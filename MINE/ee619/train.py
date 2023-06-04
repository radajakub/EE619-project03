from argparse import ArgumentParser
from collections import deque
from dm_env import TimeStep
import numpy as np

def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('-q', action='store_false', dest='log')
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-episodes', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='run')
    parser.add_argument('--test-every', type=int, default=1000)
    parser.add_argument('--test-num', type=int, default=10)
    return parser

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


def main(domain: str,
         gamma: float,
         learning_rate: float,
         log: bool,
         num_episodes: int,
         save_path: str,
         seed: int,
         task: str,
         test_every: int,
         test_num: int):
         pass

if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
