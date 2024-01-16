# from mujoco_py.generated import const
from mujoco_py import GlfwContext
import cv2
import numpy as np
import os
from Common import get_params
import torch

# GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        for z in range(self.n_skills):
            s = self.env.reset()
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            for _ in range(self.env.spec.max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r

                if done:
                    break
                s = s_

            print(f"skill: {z}, episode reward: {episode_reward:.1f}")
           
        self.env.close()

#Ant轨迹展示
    # def evaluate(self):
    #         save_path='/home/admin641/DIAYN-PyTorch-main/trajectories/'
            
    #         # 随机种子
    #         # params = get_params()
    #         # self.env.seed(params["seed"])
    #         # torch.manual_seed(params["seed"])
    #         # np.random.seed(params["seed"])

    def evaluate(self):
            save_path = '/home/admin641/DIAYN-PyTorch-main/trajectories/'

            # 为每个技能创建文件夹
            for i in range(self.n_skills):
                skill_folder = os.path.join(save_path, f'skill_{i}')
                os.makedirs(skill_folder, exist_ok=True)

                for j in range(3):
                    traj = []
                    z = i  # 设置技能值

                    s = self.env.reset()
                    initial_position = [self.env.get_body_com("torso")[0], self.env.get_body_com("torso")[1]]
                    print(initial_position)
                    traj.append(initial_position)  # 记录初始坐标
                    s = self.concat_state_latent(s, z, self.n_skills)
                    episode_reward = 0

                    for _ in range(10):
                        action = self.agent.choose_action(s)
                        s_, r, done, info = self.env.step(action)
                        s_ = self.concat_state_latent(s_, z, self.n_skills)
                        episode_reward += r
                        traj.append([info['x_position'], info['y_position']])

                        if done:
                            break
                        s = s_

                    print(f"skill: {z}, episode:{episode_reward:.1f}")

                    traj = np.array(traj)

                    file_path = os.path.join(skill_folder, f'trajectory_{j}.txt')
                    np.savetxt(file_path, traj)

            self.env.close()
