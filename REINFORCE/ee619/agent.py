"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, join, realpath
from typing import Dict, Tuple

from dm_env import TimeStep
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal


ROOT = dirname(abspath(realpath(__file__)))  # path to the directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32)


class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        self.policy = Policy(24, 6)
        self.path = join(ROOT, 'trained_model.pt')

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        observation = flatten_and_concat(time_step.observation)
        action = self.policy.act(observation)
        return np.tanh(action)

    def load(self):
        """Loads network parameters if there are any."""
        self.policy.load_state_dict(torch.load(self.path))


class Policy(nn.Module):
    """3-Layer MLP to use as a policy for DMControl environments."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.scale = nn.Parameter(torch.zeros(out_features))
        torch.nn.init.constant_(self.scale, -0.5)

    def forward(self, input_: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the location and scale for the Gaussian distribution
        to sample the action from.

        """
        loc = torch.tanh(self.fc1(input_))
        loc = torch.tanh(self.fc2(loc))
        loc = self.fc3(loc)
        scale = self.scale.exp().expand_as(loc)
        return loc, scale

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Sample an action for the given observation."""
        loc, scale = self(to_tensor(observation).unsqueeze(0))
        action = Independent(Normal(loc, scale), 1).sample().squeeze(0).numpy()
        return action
