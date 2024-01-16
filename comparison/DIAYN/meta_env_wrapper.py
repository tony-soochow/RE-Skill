import copy
from gym import Wrapper, spaces
import numpy as np
import random

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])



class meta_env_wrapper(Wrapper):
    
    def __init__(self, env, lower_level_agent, time_steps_per_skill, num_skills, states_dim):
        Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(num_skills)
        self.lower_level_agent = lower_level_agent
        self.time_steps_per_skill = time_steps_per_skill
        self.num_skills = num_skills
        self.states_dim = states_dim
    

    def reset(self, **kwargs):
        self.state = self.env.reset(**kwargs)
        self.state = np.concatenate([self.state, np.zeros(self.states_dim - self.env.observation_space.shape[0])])
        return self.state

    def step(self, skill_chosen):
        cumulative_reward = 0
        for _ in range(self.time_steps_per_skill):
            combined_state = concat_state_latent(self.state, skill_chosen, self.num_skills)
            action = self.lower_level_agent.choose_action(combined_state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.concatenate([next_state, np.zeros(self.states_dim - self.env.observation_space.shape[0])])
            cumulative_reward += reward
            self.state = next_state
            if done: break
        return next_state, cumulative_reward, done, _

