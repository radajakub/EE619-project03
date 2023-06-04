from os.path import abspath, dirname, realpath, join, isdir, basename, exists
from os import listdir, mkdir
from argparse import ArgumentParser
from collections import deque
from typing import NamedTuple
from dm_env import TimeStep
import numpy as np

from agent import flatten_and_concat, Policy, QFunction

ROOT = dirname(abspath(realpath(__file__)))  # path to the directory
EXPERIMENTS_FOLDER = 'experiments'

def default_save_path() -> str:
    parent = dirname(ROOT)
    experiments = join(parent, EXPERIMENTS_FOLDER)
    if not exists(experiments):
        mkdir(experiments)
    i = 0
    for filename in listdir(experiments):
        if isdir(filename):
            base = basename(filename)
            tmp = 0
            try:
                tmp = int(base)
            finally:
                if tmp > i:
                    i = tmp
    i += 1
    return join(experiments, str(i).zfill(2))


def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('--save_path', type=str, default=default_save_path())
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
    def add(self, state: np.ndarray, action: np.ndarray, time_step: TimeStep) -> None:
        self.states.append(flatten_and_concat(state.copy()))
        self.actions.append(action.copy())
        self.rewards.append(time_step.reward.copy())
        self.next_states.append(flatten_and_concat(time_step.observation.copy()))

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
