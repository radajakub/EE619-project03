from collections import deque
from typing import Tuple
import numpy as np

from agent import to_tensor

class Batch:
    def __init__(self, states, actions, rewards, next_states, signals):
        self.s = states
        self.a = actions
        self.r = rewards
        self.s_ = next_states
        self.d = signals

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
        temp_states = to_tensor(self.states)
        temp_actions = to_tensor(self.actions)
        temp_rewards = to_tensor(self.rewards)
        temp_next_states = to_tensor(self.next_states)
        temp_signals = to_tensor(self.dones)

        return Batch(temp_states[indices], temp_actions[indices], temp_rewards[indices], temp_next_states[indices], temp_signals[indices])
