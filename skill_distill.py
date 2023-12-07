import os.path

import gym
from Brain.agent import SACAgent
from Brain.agent_new import DISTILLAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py
from gym import Wrapper, spaces
import torch
from main import concat_state_latent
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.adam import Adam

class env_wrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self,env)
        self.env = env

    def reset(self,**kwargs):
        state = self.env.reset(**kwargs)
        self.state = np.concatenate([state, np.zeros(states_dim - env.observation_space.shape[0])])
        return self.state

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = np.concatenate([observation, np.zeros(states_dim - env.observation_space.shape[0])])
        return observation, reward, done, info



# 获得每个skill的状态集合
def get_skill_state(env, agent,n_skills, L, filename):
    S=[]
    max_episode_length = 1000
    for i in tqdm(range(n_skills)):
        s = []
        for j in range(L):
            # env.render()
            state = env.reset()
            state = concat_state_latent(state, i, n_skills)
            s.append(state)
            for t in range(max_episode_length):
                action = agent.choose_action(state)
                next_state, _, done, _ = env.step(action)
                state = next_state
                state = concat_state_latent(state, i, n_skills)
                s.append(state)

        s = np.array(s)
        S.append(s.flatten())
    S = np.array(S)
    np.savetxt(filename, S, delimiter=',')
    return S

# # 余弦相似度
# def cosine_similarity(x,y):
#     num = x.dot(y.T)
#     denom = np.linalg.norm(x) * np.linalg.norm(y)
#     return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

# # 两两对比，计算相似度
# def cal_sim(S):
#     sim_dict = {}
#     for i in range(n_skills):
#         for j in tqdm(range(i+1, n_skills)):
#             res = cosine_similarity(S[i].flatten(), S[j].flatten())
#             sim_dict[(i, j)] = res
#     #f = zip(sim_dict.keys(), sim_dict.values())
#     sorted_res = sorted(sim_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
#     return sorted_res

# 将一组相似的技能聚合为一个新技能
# skill_index为该组技能聚合后的新下标
# agent_old:教师网络，接受state与技能下标拼接作为输入，输出对应技能的action,技能数为50
# agent_new:学生网络，技能数为20
# sim_skills:一组相似技能的下标列表
# new_skill_index:一组相似技能融合到学生网络中的新下标
# k:学生网络的技能数
def distill_skill(env, agent_old, agent_new, sim_skills, new_skill_index, k):
    n_skills = 50
    max_eps = 800
    for ep in tqdm(range(max_eps)):
        state = env.reset()
        max_q_skill_index = -1
        for step in range(1, 1 + max_episode_length):

            # 初始化max_q用于比较
            tmp_state = concat_state_latent(state, 0, n_skills)  # 将原始state与选择的技能下标拼接作为网络输入
            tmp_state = np.expand_dims(tmp_state, axis=0)
            tmp_state = torch.from_numpy(tmp_state).float().to("cuda")
            reparam_actions, log_probs = agent_old.policy_network.sample_or_likelihood(tmp_state)
            max_q = torch.zeros_like(agent_old.q_value_network1(tmp_state, reparam_actions))

            # 找出多个相似技能中q值最大的技能的动作的下标
            for z in sim_skills:
                tmp_state = concat_state_latent(state, z, n_skills)
                tmp_state = np.expand_dims(tmp_state, axis=0)
                tmp_state = torch.from_numpy(tmp_state).float().to("cuda")
                reparam_actions, log_probs = agent_old.policy_network.sample_or_likelihood(tmp_state)
                q1 = agent_old.q_value_network1(tmp_state, reparam_actions)
                q2 = agent_old.q_value_network2(tmp_state, reparam_actions)
                q = torch.min(q1, q2)
                if torch.min(q, max_q) == max_q:
                    max_q_skill_index = z

            # 用q值最大的动作与环境交互
            old_state = concat_state_latent(state, max_q_skill_index, n_skills)
            action = agent_old.choose_action(old_state)
            next_state, reward, done, _ = env.step(action)

            # 将q值最大的动作的存入学生buffer
            next_state_distill = concat_state_latent(next_state, new_skill_index, k)
            state_distill = concat_state_latent(state, new_skill_index, k)
            agent_new.store(state_distill, new_skill_index, done, action, next_state_distill, old_state)

            logq_zs = agent_new.train_new_agent(agent_old)

            state = next_state
            if done:
                break
    agent_new.save()

if __name__=='__main__':
    k = 20
    n_skills = 50
    states_dim = 11
    env_name = "Hopper-v3"
    L = 100  # trajectories
    max_episode_length = 1000
    filename = "features/" + env_name + ".txt"
    print("Initializing...")
    # 初始化
    test_env = gym.make(env_name)
    test_env = env_wrapper(test_env)
    n_states = states_dim
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params_old = {"env_name": env_name,
              "n_states": n_states,
              "n_actions": n_actions,
              "action_bounds": action_bounds,
              "interval": 20,
              "do_train": False,
              "train_from_scratch": False,
              "mem_size": int(1e+6),
              "n_skills": n_skills,
              "reward_scale": float(1),
              "seed": 123,
              "lr": 3e-4,
              "batch_size": 256,
              "max_n_episodes": 5000,
              "max_episode_len": 1000,
              "gamma": 0.99,
              "alpha": 0.1,
              "tau": 0.005,
              "n_hiddens": 300}

    del test_env

    env = gym.make(env_name)
    env = env_wrapper(env)

    # initialize agent_old
    p_z_old = np.full(n_skills, 1 / n_skills)
    agent_old = SACAgent(p_z=p_z_old, **params_old)
    logger_old = Logger(agent_old, new=False, k=n_skills, **params_old)
    logger_old.load_weights()

    # initialize agent_new
    params_new = {"env_name": env_name,
                  "n_states": n_states,
                  "n_actions": n_actions,
                  "action_bounds": action_bounds,
                  "interval": 20,
                  "do_train": False,
                  "train_from_scratch": False,
                  "mem_size": int(1e+6),
                  "n_skills": k,
                  "reward_scale": float(1),
                  "seed": 123,
                  "lr": 3e-4,
                  "batch_size": 256,
                  "max_n_episodes": 5000,
                  "max_episode_len": 1000,
                  "gamma": 0.99,
                  "alpha": 0.1,
                  "tau": 0.005,
                  "n_hiddens": 300}
    p_z_new = np.full(k, 1 / k)
    agent_new = DISTILLAgent(p_z=p_z_new, new=True, k=k, **params_new)
    logger_new = Logger(agent_new, **params_new)


    # print("Getting Features...")
    # if os.path.exists(filename):
    #     S = np.loadtxt(filename, delimiter=',')
    # else:
    #     S = get_skill_state(env, agent_old, n_skills, L, filename)
    # print("Features Get")
    #sorted_sim = cal_sim(S)
    #print(sorted_sim, file=open('cal_sim.txt', 'w'))

    clusters = {1: [0, 2, 4, 7, 8, 26, 28, 29, 39, 40, 45, 46], 11: [1], 9: [3], 17: [5, 9, 10, 11, 16, 20, 23, 34, 44], 6: [6], 14: [12, 13, 15, 19, 27, 35, 37, 47], 8: [14], 10: [17, 18, 24, 25, 48], 13: [21], 18: [22], 5: [30], 16: [31], 2: [32], 19: [33], 7: [36], 3: [38], 15: [41], 0: [42], 12: [43], 4: [49]}

    #skill polymerize
    c = 0
    for skill_index, sim_skills in clusters.items():
        print("------------------------------Processing---" + str(int(c/k*100)) + "%--------------------------------------------")
        distill_skill(env, agent_old, agent_new, sim_skills, skill_index, k)  # env, agent_old, agent_new, sim_skills, skill_index, k
        c += 1
    agent_new.save()









