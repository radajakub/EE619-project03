"""Agent for DMControl Walker-Run task."""
from __future__ import annotations
from typing import Dict
from os.path import abspath, dirname, realpath
from dm_env import TimeStep
import numpy as np
import torch as th


ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory


def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.
    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])


class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        pass

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        step_type = time_step.step_type
        observation = flatten_and_concat(time_step.observation)
        reward = time_step.reward
        discount = time_step.discount
        return np.ones(6)

    def load(self):
        """Loads network parameters if there are any."""
        # Example:
        #   path = join(ROOT, 'model.pt')
        #   self.policy.load_state_dict(torch.load(path))


class QFunction:
    """
    Approximate Q function

    The network takes state as input and returns value of a given state action pair
    """
    def __init__(self, inputs: int, outputs: int) -> None:
        self.network = th.nn.Sequential(
            th.nn.Linear(inputs, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, outputs),
        )

    def clone_params(self, other: QFunction) -> None:
        self.network.load_state_dict(other.network.state_dict)

    def update_params(self, other: QFunction, rho: float) -> None:
        assert(rho >= 0 and rho <= 1)
        new_state = rho * self.network.state_dict + (1 - rho) * other.network.state_dict
        self.network.load_state_dict(new_state)

class Policy:
    def __init__(self) -> None:
        pass
