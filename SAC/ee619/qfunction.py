from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        joint_dim = state_dim + action_dim
        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q.squeeze(-1)

    def hard_update(self, other: QFunction) -> None:
        self.load_state_dict(deepcopy(other.state_dict()))

    def soft_update(self, other: QFunction, tau: float=0.005) -> None:
        for this_param, other_param in zip(self.parameters(), other.parameters()):
            this_param.data.copy_(tau * other_param.data + (1 - tau) * this_param.data)

    def clone(self, trainable=False) -> QFunction:
        new_q = QFunction(self.state_dim, self.action_dim, self.hidden_dim)
        new_q.load_state_dict(deepcopy(self.state_dict()))
        new_q.train(mode=trainable)
        return new_q
