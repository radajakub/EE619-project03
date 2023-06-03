from collections import deque
from dm_env import TimeStep
import numpy as np


class ReplayBuffer:
    def __init__(self, size: int = 1000000, batch_size: int = 100) -> None:
        assert (batch_size <= size)
        self.buffer = deque([], maxlen=size)
        self.batch_size = batch_size

    # add TimeStep into the replay buffer
    def add(self, time_step: TimeStep) -> None:
        self.buffer.append(time_step)

    def can_sample(self) -> bool:
        return len(self.buffer) >= self.batch_size

    # draw a sample of TimeSteps with the size of batch_size
    def sample(self) -> np.ndarray:
        return np.random.choice(self.buffer, size=self.batch_size, replace=False)
