from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=64) -> None:
        super().__init__()
        joint_dim = state_dim + action_dim
        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

    def hard_update(self, other: QFunction) -> None:
        self.load_state_dict(other.state_dict())

    def soft_update(self, other: QFunction, tau: float=0.005) -> None:
        for target_param, param in zip(self.parameters(), other.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
