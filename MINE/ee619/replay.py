from collections import deque
from typing import NamedTuple
import numpy as np

class Batch(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray

class ReplayBuffer:
    def __init__(self, size: int = 1000000, batch_size: int = 100) -> None:
        assert (batch_size <= size)
        self.batch_size = batch_size

        self.states = deque([], maxlen=size)
        self.actions = deque([], maxlen=size)
        self.rewards = deque([], maxlen=size)
        self.next_states = deque([], maxlen=size)

    # add TimeStep into the replay buffer
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward.copy())
        self.next_states.append(next_state.copy())

    def can_sample(self) -> bool:
        return len(self.states) >= self.batch_size

    # draw a sample of TimeSteps with the size of batch_size
    def sample(self) -> Batch:
        indices = np.random.choice(len(self.states), size=self.batch_size, replace=False)
        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices]
        )
