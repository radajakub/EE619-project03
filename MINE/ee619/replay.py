from collections import deque
from typing import Tuple
import numpy as np

class ReplayBuffer:
    def __init__(self, size: int = 1000000, batch_size: int = 100) -> None:
        assert (batch_size <= size)
        self.batch_size = batch_size
        self.size = size

        self.states = deque([], maxlen=size)
        self.actions = deque([], maxlen=size)
        self.rewards = deque([], maxlen=size)
        self.next_states = deque([], maxlen=size)
        self.dones = deque([], maxlen=size)

    # add TimeStep into the replay buffer
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, signal: int) -> None:
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward.copy())
        self.next_states.append(next_state.copy())
        self.dones.append(signal)

    def can_sample(self) -> bool:
        return len(self.states) >= self.batch_size

    # draw a sample of TimeSteps with the size of batch_size
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.states), size=self.batch_size, replace=False)
        temp_states = np.array(self.states)
        temp_actions = np.array(self.actions)
        temp_rewards = np.array(self.rewards)
        temp_next_states = np.array(self.next_states)
        temp_signals = np.array(self.dones)

        return temp_states[indices], temp_actions[indices], temp_rewards[indices], temp_next_states[indices], temp_signals[indices]
