from os import listdir, mkdir
from os.path import abspath, dirname, realpath, join, isdir, basename, exists
import numpy as np
from typing import Optional

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

class Stats:
    def __init__(self, episode_count: int, episode_length: int, save_path: str, gamma: float) -> None:
        self.rewards = np.zeros((episode_count, episode_length))
        self.save_path = join(save_path, 'rewards.npy')
        self.last_episode = 0

    def save_rewards(self, rewards: np.ndarray) -> None:
        self.rewards[self.last_episode, :] = rewards

    def save(self) -> None:
        np.save(self.save_path, self.rewards)

    def load(self, path: Optional[str]=None) -> None:
        if path is None:
            path = self.save_path
        self.rewards = np.load(path)
