import os
import gym
import uuid
import wandb
import torch
import numpy as np
from typing import Optional
from cic.utils import set_seed, rollout, ReplayBuffer
from tqdm.auto import trange, tqdm
import thop

class CICTrainer:
    def __init__(
            self,
            train_env: str,
            eval_env: Optional[str] = None,
            checkpoints_path: Optional[str] = None,
            eval_seed: int = 42,
            log_every: int = 1,
    ):
        self.train_env = gym.make(train_env)
        self.eval_env = None
        if eval_env is not None:
            self.eval_env = gym.make(eval_env)

        self.eval_seed = eval_seed

        self.log_every = log_every
        self.checkpoints_path = checkpoints_path

    @torch.no_grad()
    def evaluate(self, agent, skill, num_episodes, seed=42):
        set_seed(env=self.eval_env, seed=seed)

        returns, lens = [], []
        for _ in trange(num_episodes, desc="Evaluation", leave=False):
            total_reward, steps = rollout(self.eval_env, agent, skill)

            returns.append(total_reward)
            lens.append(steps)

        return np.array(returns), np.array(lens)

    def train(self, agent, exploration, timesteps, start_train, batch_size, buffer_size, update_skill_every, update_every, eval_every, seed=42):
        run_name = os.path.join(self.checkpoints_path, str(uuid.uuid4()))
        os.makedirs(run_name, exist_ok=True)
        print(f"Run checkpoints path: {run_name}")

        set_seed(env=self.train_env, seed=seed)

        total_loss, total_updates, total_reward = 0, 0, 0
        buffer = ReplayBuffer(agent.obs_dim, agent.action_dim, agent.skill_dim, buffer_size, device=agent.device)

        # select inital skill
        skill = agent.get_new_skill()

        done, state = False, self.train_env.reset()
        for step in trange(1, timesteps + 1, desc="Training", leave=True):
            if done:
                done, state = False, self.train_env.reset()
                # skill = agent.get_new_skill()
            elif step % update_skill_every == 0:
                skill = agent.get_new_skill()

            if step <= start_train:
                action = self.train_env.action_space.sample()
            else:
                action, exp_info = exploration(agent.act(state, skill))

            next_state, reward, done, info = self.train_env.step(action)

            is_time_limit = "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
            real_done = done and not is_time_limit

            buffer.add((state, action, reward, next_state, real_done, skill))

            state = next_state
            total_reward += reward

            exploration.step()
            if step > start_train:
                if step % update_every:
                    batch = buffer.sample(size=batch_size)
                    update_info = agent.update(batch)
                    # break
                    total_loss += update_info["loss"]
                    total_updates += 1

                    if step % self.log_every == 0:
                        wandb.log({
                            "step": step,
                            "reward_mean": total_reward / step,
                            **update_info,
                            **exp_info
                        })

                if step % eval_every == 0:
                    if self.eval_env is not None:
                        agent.eval()
                        returns, lens = self.evaluate(agent, skill, num_episodes=25, seed=self.eval_seed)
                        agent.train()

                        wandb.log({
                            "step": step,
                            "eval/reward_mean": np.mean(returns),
                            "eval/reward_std": np.std(returns),
                            "eval/mean_average_reward": np.mean(returns / lens)
                        })

                    torch.save(agent, os.path.join(run_name, f"agent_{step}.pt"))
                    tqdm.write(f"Step: {step}, Loss: {total_loss / total_updates}, Intristic Reward: {update_info['int_reward_batch_mean']}"
                               f", Reward mean: {total_reward / step}")