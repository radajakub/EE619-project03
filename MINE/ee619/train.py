from os.path import dirname, abspath, realpath, join
from datetime import datetime
from functools import reduce
import operator
from argparse import ArgumentParser
from typing import Iterable, List, Optional
from dm_control import suite
from dm_control.rl.control import Environment
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import trange

from agent import flatten_and_concat, GaussianPolicy, to_tensor
from replay import ReplayBuffer
from qfunction import QFunction
from temperature import ConstAlpha, AutotuningAlpha

ROOT = dirname(abspath(realpath(__file__)))  # path to the directory

def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-episodes', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', default='run')
    parser.add_argument('--test-every', type=int, default=1000)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=None)
    return parser

def main(domain: str,
         gamma: float,
         tau: float,
         learning_rate: float,
         num_episodes: int,
         seed: int,
         task: str,
         test_every: int,
         test_num: int,
         temperature: Optional[float]):


    print(f'===== HYPERPARAMTERS =====')
    print(f'Domain and task: {domain} - {task}')
    print(f'Discount factor gamma: {gamma}')
    print(f'Target update weight tau: {tau}')
    print(f'Learning rate: {learning_rate}')
    print(f'Number of episodes: {num_episodes}')
    print(f'Random seed: {seed}')
    print(f'Temperature alpha: {temperature if temperature is not None else "automatic"}')

    # init seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setup logging utility
    writer = SummaryWriter(join(dirname(ROOT), 'experiments'))

    # init environment and query information
    env: Environment = suite.load(domain, task, task_kwargs={'random': seed})
    observation_spec = env.observation_spec()
    state_shape = np.sum([np.prod(value.shape, dtype=int) for value in observation_spec.values()])
    action_spec = env.action_spec()
    action_shape = np.prod(action_spec.shape, dtype=int)
    max_action = action_spec.maximum
    min_action = action_spec.minimum
    action_loc = (max_action + min_action) / 2
    action_scale = (max_action - min_action) / 2

    # replay buffer to store seen samples from the environment
    replay_buffer = ReplayBuffer()

    # define temperature class
    alpha = AutotuningAlpha(action_shape, learning_rate) if temperature is None else ConstAlpha(temperature)

    # define Q networks
    Q1 = QFunction(state_shape, action_shape)
    Q1.train()
    Q1_optim = Adam(Q1.parameters(), lr=learning_rate)
    Q2 = QFunction(state_shape, action_shape)
    Q2.train()
    Q2_optim = Adam(Q2.parameters(), lr=learning_rate)

    # define target Q networks and clone the parameters
    Q1_target = QFunction(state_shape, action_shape)
    Q1_target.hard_update(Q1)
    Q2_target = QFunction(state_shape, action_shape)
    Q2_target.hard_update(Q2)

    # initialize gaussian policy and set it to train mode
    pi = GaussianPolicy(state_shape, action_shape, action_loc=action_loc, action_scale=action_scale)
    pi.train()
    pi_optim = Adam(pi.parameters(), lr=learning_rate)

    updates = 0
    step_count = 0

    for episode in range(num_episodes):
        # start new episode in the environment
        time_step = env.reset()
        episode_rewards: List[float] = []

        # rollout episode
        while not time_step.last():
            # make a step
            state = flatten_and_concat(time_step.observation)
            with torch.no_grad():
                action = pi.act(to_tensor(state))
            time_step = env.step(action)

            # save transition to replay buffer
            replay_buffer.push(state, action, time_step.reward, flatten_and_concat(time_step.observation))

            step_count += 1
            episode_rewards.append(time_step.reward)

            # update weights
            if replay_buffer.can_sample():
                # sample from replay buffer and convert to tensors
                s, a, r, s_ = replay_buffer.sample()
                s = to_tensor(s)
                a = to_tensor(a)
                r = to_tensor(r)
                s_ = to_tensor(s_)

                # update Q functions
                # compute targets
                with torch.no_grad():
                    locs, scales = pi(s_)
                    a_, log_probs = pi.act_with_log_probs(locs, scales)
                    min_q = torch.minimum(Q1_target(s_, a_), Q2_target(s_, a_)).squeeze(1)
                    target = r + gamma * (min_q - alpha.get() * log_probs)
                    target = target.unsqueeze(1)

                Q1_optim.zero_grad()
                Q1_loss = F.mse_loss(Q1(s, a), target.detach())
                Q1_loss.backward()
                Q1_optim.step()

                Q2_optim.zero_grad()
                Q2_loss = F.mse_loss(Q2(s, a), target.detach())
                Q2_loss.backward()
                Q2_optim.step()

                # update policy pi
                locs, scales = pi(s)
                at, at_log_probs = pi.act_with_log_probs(locs, scales)
                min_q = torch.minimum(Q1(s, at), Q2(s, at)).squeeze(1)

                pi_optim.zero_grad()
                pi_loss = ((alpha.get() * at_log_probs) - min_q).mean()
                pi_loss.backward()
                pi_optim.step()

                alpha.update(at_log_probs)

                # update target Q functions
                Q1_target.soft_update(Q1, tau=tau)
                Q2_target.soft_update(Q2, tau=tau)

                # save losses
                writer.add_scalar('loss/Q1', Q1_loss, updates)
                writer.add_scalar('loss/Q2', Q2_loss, updates)
                writer.add_scalar('loss/pi', pi_loss, updates)
                writer.add_scalar('temperature', alpha.get(), updates)
                updates += 1

        # episode_return = np.dot(gammas, np.array(episode_rewards))
        episode_return = np.sum(episode_rewards)
        writer.add_scalar('return/train', episode_return, episode)
        print(f"Episode: {episode}, steps: {step_count}, return: {round(episode_return, 2)}")


        # testing
        if episode % test_every == 0:
            returns = []
            for _ in range(test_num):
                rewards = []
                time_step = env.reset()
                while not time_step.last():
                    pi.eval()
                    action = pi.act(to_tensor(flatten_and_concat(time_step.observation)))
                    pi.train()
                    time_step = env.step(action)
                    rewards.append(time_step.reward)
                returns.append(np.sum(episode_rewards))
            avg_return = np.average(returns)
            writer.add_scalar('average_return/test', avg_return, episode)
            print(f"Test in episode: {episode}, average return: {round(avg_return, 2)}")

    env.close()

    torch.save(pi.state_dict(), 'trained_policy.pt')


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
