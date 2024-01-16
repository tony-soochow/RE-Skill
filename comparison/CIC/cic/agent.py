import math
import torch
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from torch.optim import Adam

from cic.networks import Actor, Critic, CIC
from cic.utils import soft_update_params, compute_apt_reward, RMS
from thop import profile


class CICAgent:
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dim: int,
            # cic parameters
            skill_dim: int = 10,
            project_skill: bool = True,
            temperature: float = 1.0,
            # td3 params
            gamma: float = 0.99,
            update_actor_every: int = 1,
            target_tau: float = 1e-3,
            noise_clip: float = 0.3,
            noise_std: float = 0.2,
            learning_rate: float = 1e-3,
            max_action: float = 1.0,
            device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.skill_dim = skill_dim
        self.total_obs_dim = obs_dim + skill_dim

        self.temperature = temperature
        self.update_actor_every = update_actor_every
        self.gamma = gamma
        self.target_tau = target_tau
        self.noise_clip = noise_clip
        self.noise_std = noise_std
        self.max_action = max_action
        self.device = device

        # models
        self.actor = Actor(self.total_obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(self.total_obs_dim, action_dim, hidden_dim).to(device)

        with torch.no_grad():
            self.actor_target = deepcopy(self.actor)
            self.critic_target = deepcopy(self.critic)
            self.reward_rms = RMS(epsilon=1e-4, shape=(1,), device=self.device)

        self.cic = CIC(obs_dim, skill_dim, hidden_dim, project_skill).to(device)

        # optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = Adam(self.critic.parameters(), lr=learning_rate)
        self.cic_opt = Adam(self.cic.parameters(), lr=learning_rate)

        self._num_updates = 0.0
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.actor_target.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        self.cic.train(training)

    def eval(self):
        self.train(training=False)

    @torch.no_grad()
    def _compute_apt_reward(self, obs, next_obs):
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(
            source,
            target,
            self.reward_rms,
            knn_k=16,
            knn_avg=True,
            use_rms=True,
            knn_clip=0.0005
        )
        return reward.unsqueeze(-1)

    def _critic_loss(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            next_action = next_action + torch.clamp(self.noise_std * torch.randn_like(next_action), -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action, -self.max_action, self.max_action)

            Q_next = torch.minimum(*self.critic_target(next_obs, next_action))
            assert reward.shape == done.unsqueeze(-1).shape == Q_next.shape
            Q_target = reward + (1 - done).unsqueeze(-1) * self.gamma * Q_next

        Q1, Q2 = self.critic(obs, action)
        # print("Critic loss", Q1.shape, Q2.shape, Q_target.shape)
        assert Q1.shape == Q2.shape == Q_target.shape
        loss = F.mse_loss(Q1, Q_target) + F.mse_loss(Q2, Q_target)

        return loss

    def _actor_loss(self, obs):
        Q1, Q2 = self.critic(obs, self.actor(obs))
        assert Q1.shape == Q2.shape
        loss = -torch.minimum(Q1, Q2).mean()
        return loss

    def _cic_loss(self, obs, next_obs, skill):
        # loss from https://github.com/rll-research/cic/blob/b523c3884256346cb585bf06e52a7aadc127dcfc/agent/cic.py#L155
        query, key = self.cic(obs, next_obs, skill)

        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        cov = torch.mm(query, key.T)
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        row_sub = torch.zeros_like(neg).fill_(math.e**(1 / self.temperature))
        neg = torch.clamp(neg - row_sub, min=1e-6)  # clamp for numerical stability
        pos = torch.exp(torch.sum(query * key, dim=-1) / self.temperature)
        loss = -torch.log(pos / (neg + 1e-6)).mean()

        return loss

    def update(self, batch):
        obs, action, ext_reward, next_obs, done, skill = [b.to(self.device) for b in batch]
        

        
        # update CIC
        cic_loss = self._cic_loss(obs, next_obs, skill)

        # flops_policy_3, params_policy_3 = profile(self.cic, inputs = (obs, next_obs, skill,))
        self.cic_opt.zero_grad()

        cic_loss.backward()
        self.cic_opt.step()

        # compute intristic reward
        int_reward = self._compute_apt_reward(next_obs, next_obs)
        # or that? In paper this it the formula, but in code they use the next_obs instead of obs
        # int_reward = self._compute_apt_reward(obs, next_obs)

        # augment state with skill
        obs = torch.cat([obs, skill], dim=-1)
        next_obs = torch.cat([next_obs, skill], dim=-1)

        #计算FLOPs 和 Params(M)
        # flops_policy_1, params_policy_1 = profile(self.actor, inputs = (obs,))
        # flops_policy_2, params_policy_2 = profile(self.critic, inputs = (obs, action,))
        
        # flops_policy = flops_policy_1 + flops_policy_2 + flops_policy_3 
        # params_policy = params_policy_1 + params_policy_2 + params_policy_3 
        # print(flops_policy, params_policy)
        # print("%s | %s | %s" % ("CIC Model", "FLOPs(G)", "Params(M)"))
        # print("------|-----------|------")
        # print("%s | %.7f | %.7f" % ("模型  ", flops_policy / (1000 ** 3), params_policy / (1000 ** 2)))

        # update Critic
        critic_loss = self._critic_loss(obs, action, int_reward, next_obs, done)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update Actor
        actor_loss = self._actor_loss(obs)
        if self._num_updates % self.update_actor_every == 0:
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            with torch.no_grad():
                soft_update_params(self.actor_target, self.actor, self.target_tau)
                soft_update_params(self.critic_target, self.critic, self.target_tau)

        self._num_updates += 1



        return {
            "loss": (critic_loss + actor_loss + cic_loss).item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "cic_loss": cic_loss.item(),
            "int_reward_batch_mean": int_reward.mean().item()
        }

    @torch.no_grad()
    def act(self, obs: np.ndarray, skill: np.ndarray):
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        skill = torch.tensor(skill, dtype=torch.float, device=self.device)
        obs = torch.cat([obs, skill], dim=-1)

        return self.actor(obs).cpu().numpy() * self.max_action

    def get_new_skill(self):
        return np.random.uniform(0, 1, self.skill_dim).astype(float)
