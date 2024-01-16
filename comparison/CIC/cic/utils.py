import os
import torch
import random
import imageio
import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Tuple, Dict, Any
from torch.nn import Module


def set_seed(env=None, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def soft_update_params(target: Module, source: Module, tau: float = 0.001):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def three_layer_mlp(input_dim: int, out_dim: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )


def weight_init(m: Module):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# from https://github.com/rll-research/cic/blob/b523c3884256346cb585bf06e52a7aadc127dcfc/agent/cic.py#L56
class RMS:
    def __init__(self, epsilon=1e-4, shape=(1,), device="cpu"):
        self.M = torch.zeros(shape, requires_grad=False).to(device)
        self.S = torch.ones(shape, requires_grad=False).to(device)
        self.n = epsilon

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


def pairwise_dist(X: Tensor, Y: Tensor):
    dists = torch.sum(X**2, dim=1, keepdims=True) + torch.sum(Y ** 2, dim=1) - 2 * X @ Y.T
    return torch.sqrt(dists)


def compute_apt_reward(
        source: Tensor,
        target: Tensor,
        rms: RMS,
        knn_k: int = 16,
        knn_avg: bool = True,
        use_rms: bool = True,
        knn_clip: float = 0.0005
) -> Tensor:
    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)

    # It will eat up all memory and 15x slower (from original code)
    # sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

    # corrected version (identical to the original, but a lot faster)
    sim_matrix = pairwise_dist(source, target)
    reward, _ = sim_matrix.topk(knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if use_rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - knn_clip, torch.zeros_like(reward).to(source.device))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if use_rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - knn_clip, torch.zeros_like(reward).to(source.device))
        reward = reward.reshape((b1, knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, skill_dim, buffer_size, device="cpu"):
        self.state = torch.zeros(buffer_size, state_dim, dtype=torch.float, device=device)
        self.action = torch.zeros(buffer_size, action_dim, dtype=torch.float, device=device)
        self.reward = torch.zeros(buffer_size, dtype=torch.float, device=device)
        self.next_state = torch.zeros(buffer_size, state_dim, dtype=torch.float, device=device)
        self.done = torch.zeros(buffer_size, dtype=torch.int, device=device)
        self.skill = torch.zeros(buffer_size, skill_dim, dtype=torch.float, device=device)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.device = device

    def add(self, transition):
        state, action, reward, next_state, done, skill = transition

        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state, device=self.device)
        self.action[self.count] = torch.as_tensor(action, device=self.device)
        self.reward[self.count] = torch.as_tensor(reward, device=self.device)
        self.next_state[self.count] = torch.as_tensor(next_state, device=self.device)
        self.done[self.count] = torch.as_tensor(done, device=self.device)
        self.skill[self.count] = torch.as_tensor(skill, device=self.device)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, size):
        assert self.real_size >= size

        sample_idxs = np.random.choice(self.real_size, size)
        batch = (
            self.state[sample_idxs],
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.done[sample_idxs],
            self.skill[sample_idxs]
        )
        return batch


# exploration scheduler
class NormalNoise:
    def __init__(self, action_dim, timesteps, max_action, eps_min, eps_max):
        self.action_dim = action_dim
        self.max_action = max_action

        self.eps_min = eps_min
        self.eps_max = eps_max
        self.timesteps = timesteps

        self._step = 0.0

    def __call__(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        eps = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * self._step / self.timesteps)
        action = np.clip(action + np.random.randn(*action.shape), -self.max_action, self.max_action)

        return action, {"noise_eps": eps}

    def step(self):
        self._step = self._step + 1


def rollout(env, agent, skill, render_path=None, max_steps=float("inf")):
    done, state = False, env.reset()

    steps, total_reward, images = 0.0, 0.0, []
    while not done:
        action = agent.act(state, skill)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if render_path is not None:
            images.append(env.render(mode="rgb_array"))

        if steps > max_steps:
            done = True

    if render_path is not None:
        imageio.mimsave(render_path, images, fps=32)

    return total_reward, steps


if __name__ == "__main__":
    torch.manual_seed(32)

    source = torch.randn(1024, 10)
    target = torch.randn(1024, 10)

    b1, b2 = source.size(0), target.size(0)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    sim_matrix_fast = pairwise_dist(source, target)

    print((sim_matrix.topk(4, dim=1, largest=False, sorted=True)[1] - sim_matrix_fast.topk(4, dim=1, largest=False, sorted=True)[1]).sum())
