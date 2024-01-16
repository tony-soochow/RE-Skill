import gym
import torch
import wandb
import numpy as np
import thop
from cic.agent import CICAgent
from cic.trainer import CICTrainer
from cic.utils import set_seed, NormalNoise, rollout

DEVICE = "cuda:0"

def main():
    set_seed(seed=32)
    env_name = "Hopper-v3"
    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    wandb.init(
        project="CIC",
        group="hopper",
        name="first_run",
        entity="df",
        mode="disabled"
    )
    agent = CICAgent(
        obs_dim=11,
        action_dim=action_dim,
        skill_dim=50, # 64
        hidden_dim=256,  # 256
        learning_rate=1e-4,
        target_tau=1e-4, # 1e-4
        update_actor_every=1,
        device=DEVICE
    )
    exploration = NormalNoise(
        action_dim=action_dim,
        timesteps=2_000_000,
        max_action=1.0,
        eps_max=0.6, # 0.6
        eps_min=0.05
    )
    trainer = CICTrainer(
        train_env=env_name,
        eval_env=env_name,
        checkpoints_path="cic_checkpoints"
    )
    trainer.train(
        agent=agent,
        exploration=exploration,
        timesteps=2_000_000,
        start_train=4000, # 4000
        batch_size=2048, # 1024
        buffer_size=1_000_000,
        update_skill_every=100, # 100
        update_every=2, # 1
        eval_every=25_000
    )

    # agent = torch.load("cic_checkpoints/45695034-daec-440f-b4cf-d505ffd1b251/agent_250000.pt")
    # skills = np.linspace(0.0, 1.0, 10)
    #
    # for i, skill_value in enumerate(skills):
    #     env = gym.make("HalfCheetah-v3")
    #     set_seed(env=env, seed=32)
    #     skill = np.zeros(agent.skill_dim) + skill_value

        # rollout(env, agent, skill, render_path=f"videos/rollout_{i}.mp4", max_steps=100)

    

if __name__ == "__main__":
    main()