import torch
from torch import nn
import torch.nn.functional as F

class VFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        v = F.relu(self.fc1(state))
        v = F.relu(self.fc2(v))
        v = self.fc3
        return v

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64) -> None:
        super().__init__()
        joint_dim = state_dim + action_dim
        self.fc1 = nn.Linear(joint_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3
        return q
