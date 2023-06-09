from os.path import dirname, abspath, realpath, join
from functools import reduce
from argparse import ArgumentParser
from typing import List, Optional
from dm_control import suite
from dm_control.rl.control import Environment
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.distributions import Independent, Normal

from agent import flatten_and_concat, GaussianPolicyLayered, GaussianPolicyParametrized, to_tensor
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
    parser.add_argument('--target-update', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--num-episodes', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', default='run')
    parser.add_argument('--test-every', type=int, default=int(1e2))
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--replay-size', type=int, default=int(1e6))
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-best', type=bool, default=True)
    parser.add_argument('--q-hidden', type=int, default=256)
    parser.add_argument('--pi-hidden', type=int, default=256)
    parser.add_argument('--pi-nonlinearity', type=str, default='tanh')
    parser.add_argument('--pi-type', type=str, default='layered')
    parser.add_argument('--save-interval', type=int, default=10000)
    return parser

def main(domain: str,
         gamma: float,
         tau: float,
         target_update: int,
         learning_rate: float,
         num_episodes: int,
         seed: int,
         task: str,
         test_every: int,
         test_num: int,
         temperature: Optional[float],
         replay_size: int,
         batch_size: int,
         save_best: bool,
         q_hidden: int,
         pi_hidden: int,
         pi_nonlinearity: str,
         pi_type: str,
         save_interval: int):

    assert (pi_type in ['layered', 'parametrized'])

    print(f'===== HYPERPARAMTERS =====')
    print(f'Domain and task: {domain} - {task}')
    print(f'Discount factor gamma: {gamma}')
    print(f'Target update weight tau: {tau}')
    print(f'Target update every: {target_update} steps')
    print(f'Learning rate: {learning_rate}')
    print(f'Number of episodes: {num_episodes}')
    print(f'Random seed: {seed}')
    print(f'Temperature alpha: {temperature if temperature is not None else "automatic"}')
    print(f'Replay buffer size: {replay_size}')
    print(f'Replay buffer batch size: {batch_size}')
    print(f'Save best: {save_best}')
    print(f'Q hidden neurons: {q_hidden}')
    print(f'PI hidden neurons: {pi_hidden}')
    print(f'PI non-linearity: {pi_nonlinearity}')
    print(f'Every {test_every} steps perform {test_num} tests')
    print(f'Save policy every {save_interval} steps')

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

    def map_action(input_: np.ndarray) -> np.ndarray:
        return np.tanh(input_) * action_scale + action_loc

    # replay buffer to store seen samples from the environment
    replay_buffer = ReplayBuffer(size=replay_size, batch_size=batch_size)

    # define temperature class
    alpha = AutotuningAlpha(action_shape, learning_rate) if temperature is None else ConstAlpha(temperature)

    # define Q networks
    Q1 = QFunction(state_shape, action_shape, hidden_dim=q_hidden)
    Q1.train()
    Q1_optim = Adam(Q1.parameters(), lr=learning_rate)
    Q2 = QFunction(state_shape, action_shape, hidden_dim=q_hidden)
    Q2.train()
    Q2_optim = Adam(Q2.parameters(), lr=learning_rate)

    # define target Q networks and clone the parameters
    Q1_target = QFunction(state_shape, action_shape, hidden_dim=q_hidden)
    Q1_target.hard_update(Q1)
    Q2_target = QFunction(state_shape, action_shape, hidden_dim=q_hidden)
    Q2_target.hard_update(Q2)

    # initialize gaussian policy and set it to train mode
    if pi_type == 'layered':
        pi = GaussianPolicyLayered(state_shape, action_shape, hidden_dim=pi_hidden, nonlinearity=pi_nonlinearity)
    else:
        pi = GaussianPolicyParametrized(state_shape, action_shape, hidden_dim=pi_hidden, nonlinearity=pi_nonlinearity)

    pi.train()
    pi_optim = Adam(pi.parameters(), lr=learning_rate)

    updates = 0

    best_return = -1
    if pi_type == 'layered':
        best_pi = GaussianPolicyLayered(state_shape, action_shape, hidden_dim=pi_hidden, nonlinearity=pi_nonlinearity)
    else:
        best_pi = GaussianPolicyParametrized(state_shape, action_shape, hidden_dim=pi_hidden, nonlinearity=pi_nonlinearity)

    for episode in range(num_episodes):
        # start new episode in the environment
        time_step = env.reset()
        episode_return = 0

        # rollout episode
        while not time_step.last():
            # make a step
            state = flatten_and_concat(time_step.observation)
            with torch.no_grad():
                action = pi.act(to_tensor(state))
            time_step = env.step(map_action(action.copy()))

            # save transition to replay buffer
            replay_buffer.push(state, action, time_step.reward, flatten_and_concat(time_step.observation))

            episode_return += time_step.reward

        # update weights
        if replay_buffer.can_sample():
            # sample from replay buffer and convert to tensors
            batch_s, batch_a, batch_r, batch_s_ = replay_buffer.sample()
            batch_s = to_tensor(batch_s)
            batch_a = to_tensor(batch_a)
            batch_r = to_tensor(batch_r)
            batch_s_ = to_tensor(batch_s_)

            # update Q functions
            # compute targets
            with torch.no_grad():
                locs, scales = pi(batch_s_)
                a_ = pi.sample(locs, scales)
                distribution = Independent(Normal(locs, scales), 1)
                a_, log_probs = pi.squash(a_, distribution.log_prob(a_))
                min_q = torch.minimum(Q1_target(batch_s_, a_), Q2_target(batch_s_, a_)).squeeze(1)
                target = batch_r + gamma * (min_q - alpha.get() * log_probs).unsqueeze(1)

            Q1_optim.zero_grad()
            Q1_loss = ((Q1(batch_s, batch_a) - target).pow(2)).mean()
            Q1_loss.backward()
            Q1_optim.step()

            Q2_optim.zero_grad()
            Q2_loss = ((Q2(batch_s, batch_a) - target).pow(2)).mean()
            Q2_loss.backward()
            Q2_optim.step()

            # update policy pi
            locs, scales = pi(batch_s)
            # rsample should be the reparametrization trick
            a1 = pi.rsample(locs, scales)
            distribution = Independent(Normal(locs, scales), 1)
            a1, a1_log_probs = pi.squash(a1, distribution.log_prob(a1))
            min_q = torch.minimum(Q1(batch_s, a1), Q2(batch_s, a1)).squeeze(1)

            pi_optim.zero_grad()
            pi_loss = -(min_q - alpha.get() * a1_log_probs).mean()
            pi_loss.backward()
            pi_optim.step()

            # update alpha parameter
            if temperature is None:
                locs, scales = pi(batch_s)
                a2 = pi.sample(locs, scales)
                distribution = Independent(Normal(locs, scales), 1)
                a2, at_log_probs = pi.squash(a2, distribution.log_prob(a2))
                alpha.update(at_log_probs)

            # update target Q functions
            if episode % target_update == 0:
                Q1_target.soft_update(Q1, tau=tau)
                Q2_target.soft_update(Q2, tau=tau)

            # save losses
            writer.add_scalar('loss/Q1', Q1_loss, updates)
            writer.add_scalar('loss/Q2', Q2_loss, updates)
            writer.add_scalar('loss/pi', pi_loss, updates)
            writer.add_scalar('temperature', alpha.get(), updates)
            updates += 1

        # episode_return = np.dot(gammas, np.array(episode_rewards))
        writer.add_scalar('return/train', episode_return, episode)
        print(f"Episode: {episode}, return: {round(episode_return, 2)}")

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
                    time_step = env.step(map_action(action))
                    rewards.append(time_step.reward)
                returns.append(np.sum(rewards))
            avg_return = np.average(returns)

            if save_best and avg_return < best_return:
                best_return = avg_return
                best_pi.hard_update(pi)

            writer.add_scalar('average_return/test', avg_return, episode)
            print(f"Test in episode: {episode}, average return: {round(avg_return, 2)}")

        if episode % save_interval == 0 and episode != 0:
            torch.save(pi.state_dict(), f'trained_model_{episode}.pt')


    env.close()

    torch.save(pi.state_dict(), 'trained_model.pt')
    if save_best:
        torch.save(best_pi.state_dict, 'traind_model_best.pt')


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
