from functools import reduce
import operator
from argparse import ArgumentParser
from typing import Iterable, List
from dm_control import suite
from dm_control.rl.control import Environment
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import trange

from agent import flatten_and_concat, GaussianPolicy, to_tensor
from replay import ReplayBuffer
from output import Stats, default_save_path
from qfunction import QFunction


def prod(iterable: Iterable[int]) -> int:
    return reduce(operator.mul, iterable, 1)


def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('--save_path', type=str, default=default_save_path())
    parser.add_argument('-q', action='store_false', dest='log')
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-episodes', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='run')
    parser.add_argument('--test-every', type=int, default=1000)
    parser.add_argument('--test-num', type=int, default=10)
    return parser

def run_episode(env: Environment, policy: GaussianPolicy, replay_buffer: ReplayBuffer, stats: Stats):
    # reset the environment (obtain start state)
    time_step = env.reset()
    # go through every step and add them to replay buffer
    rewards: List[float] = []
    while not time_step.last():
        state = flatten_and_concat(time_step.observation)
        # do not compute gradient
        with torch.no_grad():
            action, _ = policy.act(to_tensor(state))
        time_step = env.step(action)
        # replay buffer takes state s_t action a_t and time_step with reward r_t+1 and next state s_t+1
        replay_buffer.push(state, action, time_step.reward, flatten_and_concat(time_step.observation))
        rewards.append(time_step.reward)
    # save rewards to stats
    stats.save_rewards(np.array(rewards))


def main(domain: str,
         gamma: float,
         tau: float,
         learning_rate: float,
         log: bool,
         num_episodes: int,
         save_path: str,
         seed: int,
         task: str,
         test_every: int,
         test_num: int):
    # init seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setup logging utility
    writer = SummaryWriter() if log else None
    stats = Stats(num_episodes, 1000, save_path, gamma)

    # init environment and query information
    env: Environment = suite.load(domain, task, task_kwargs={'random': seed})
    observation_spec = env.observation_spec()
    state_shape = np.sum([prod(value.shape) for value in observation_spec.values()])
    action_spec = env.action_spec()
    action_shape = prod(action_spec.shape)
    max_action = action_spec.maximum
    min_action = action_spec.minimum
    action_loc = (max_action + min_action) / 2
    action_scale = (max_action - min_action) / 2

    replay_buffer = ReplayBuffer()

    # define Q networks
    Qnum = 2
    Qs = [QFunction(state_shape, action_shape) for _ in range(Qnum)]
    for Q in Qs:
        Q.train()
    Q_optims = [Adam(Qs[i].parameters(), lr=learning_rate) for i in range(Qnum)]

    # define target Q networks and clone the parameters
    Qtargets = [QFunction(state_shape, action_shape) for _ in range(Qnum)]
    for i in range(Qnum):
        Qtargets[i].hard_update(Qs[i])
    # optimizer is not needed as the parameters are updated from the main Q networks

    # initialize gaussian policy and set it to train mode
    pi = GaussianPolicy(state_shape, action_shape, action_loc, action_scale)
    pi.train()
    pi_optim = Adam(pi.parameters(), lr=learning_rate)

    for episode in trange(num_episodes):
        run_episode(env, pi, replay_buffer, stats)

        if not replay_buffer.can_sample():
            continue

        # sample batch from replay buffer
        states, actions, rewards, next_states = replay_buffer.sample()

        # update Q functions
        # sample next action

        # update policy pi

        # update target V function
        for i in range(Qnum):
            Qtargets[i].soft_update(Qs[i], tau=tau)


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
