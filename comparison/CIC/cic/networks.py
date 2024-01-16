import torch
import torch.nn as nn

from cic.utils import weight_init, three_layer_mlp


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        action = torch.tanh(self.policy(h))
        return action


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.Q1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        inpt = torch.cat([obs, action], dim=-1)
        h = self.trunk(inpt)

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = three_layer_mlp(self.obs_dim, self.skill_dim, hidden_dim)
        self.next_state_net = three_layer_mlp(self.obs_dim, self.skill_dim, hidden_dim)
        self.pred_net = three_layer_mlp(2 * self.skill_dim, self.skill_dim, hidden_dim)

        if project_skill:
            self.skill_net = three_layer_mlp(self.skill_dim, self.skill_dim, hidden_dim)
        else:
            self.skill_net = nn.Identity()

        self.apply(weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())

        state = self.state_net(state)
        next_state = self.state_net(next_state)

        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state], dim=1))

        return query, key
