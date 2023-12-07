import gym
from Brain import SACAgent
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
import os
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

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
    max_episode_length = 100
    for i in tqdm(range(n_skills)):
        s = []
        for j in range(L):
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

# 获得每个skill的动作集合
def get_skill_action(env, agent,n_skills, L, filename):
    A=[]
    max_episode_length = 1000
    for i in tqdm(range(n_skills)):
        a = []
        for j in range(L):
            # env.render()
            state = env.reset()
            state = concat_state_latent(state, i, n_skills)
            for t in range(max_episode_length):
                action = agent.choose_action(state)
                a.append(action)
                next_state, _, done, _ = env.step(action)
                state = next_state
                state = concat_state_latent(state, i, n_skills)

        a = np.array(a)
        A.append(a.flatten())
    A = np.array(A)
    np.savetxt(filename, A, delimiter=',')
    return A



if __name__=='__main__':
    n_skills = 50
    states_dim = 11
    env_name = "Hopper-v3"
    L = 150  # trajectories
    max_episode_length = 100
    filename = "features/" + env_name + "-" + str(n_skills) + ".txt"
    print("Initializing...")
    # 初始化
    test_env = gym.make(env_name)
    test_env = env_wrapper(test_env)
    n_states = states_dim
    n_actions = test_env.action_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

    params = {"env_name": env_name,
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

    # del test_env, n_states, n_actions, action_bounds

    env = gym.make(env_name)
    env = env_wrapper(env)
    state = env.reset()
    p_z = np.full(n_skills, 1 / n_skills)
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)
    logger.load_weights()
    print("Initializing Success")
    print("Getting Features...")
    if os.path.exists(filename):
        S = np.loadtxt(filename, delimiter=',')
    else:
        S = get_skill_state(env, agent, n_skills, L, filename)
    print("Features Get")

    # print("Clustering...")
    # num = range(1, 80)  # 分别模拟k的情况
    # sse_result = []  # 用于存放每种k聚类后的SSE
    # for k in tqdm(num):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(S)
    #     sse_result.append(kmeans.inertia_)  # inertia_表示样本到最近的聚类中心的距离总和。    
    # plt.plot(num, sse_result, 'b*:')  # 'b*:'为线的格式设置，b表示蓝色，*为点的标记，:表示线型为点状线条�
    # plt.show()

    #skill cluster
    k = 20
    print("Skill clustering...")
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(S)
    results = kmeans.predict(np.array(S))
    print("Skill cluster results:")
    print('results:',results)
    
    silhouette_avg = silhouette_score(S, results)
    print(f"Silhouette Coefficient: {silhouette_avg}")

    davies_bouldin_index = davies_bouldin_score(S, results)
    print(f"Davies-Bouldin Index: {davies_bouldin_index}")

    calinski_harabasz_index = calinski_harabasz_score(S, results)
    print(f"Calinski-Harabasz Index: {calinski_harabasz_index}")

    clusters = {}
    for i in range(len(results)):
      if results[i] not in clusters:
          t = []
          t.append(i)
          clusters[results[i]] = t
      else:
          clusters[results[i]].append(i)
    print('clusters:',clusters)

    # x = [i[0] for i in S]
    # y = [i[1] for i in S]
    # plt.scatter(x, y, c=results, marker='o')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

