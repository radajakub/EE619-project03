from os.path import dirname, abspath, realpath, join
from argparse import ArgumentParser
from typing import Optional
from dm_control import suite
from dm_control.rl.control import Environment
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import flatten_and_concat, to_tensor
from replay import ReplayBuffer
from sac import SAC

ROOT = dirname(abspath(realpath(__file__)))  # path to the directory

DM_ROLLOUT_LENGTH = 1000

def build_argument_parser() -> ArgumentParser:
    """Returns an argument parser for main."""
    parser = ArgumentParser()
    parser.add_argument('--domain', default='walker')
    parser.add_argument('--task', default='run')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--target-update', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--update-every', type=int, default=50)
    parser.add_argument('--num-episodes', type=int, default=int(1000))
    parser.add_argument('--steps-per-episode', type=int, default=int(1e4))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--replay-size', type=int, default=int(1e6))
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-intermediate', type=bool, default=True)
    parser.add_argument('--q-hidden', type=int, default=256)
    parser.add_argument('--pi-hidden', type=int, default=256)
    parser.add_argument('--pi-nonlinearity', type=str, default='relu')
    parser.add_argument('--initial-steps', type=int, default=10000)
    return parser

def main(domain: str,
         task: str,
         gamma: float,
         tau: float,
         target_update: int,
         learning_rate: float,
         update_every: int,
         num_episodes: int,
         steps_per_episode: int,
         seed: int,
         test_num: int,
         temperature: Optional[float],
         replay_size: int,
         batch_size: int,
         save_intermediate: bool,
         q_hidden: int,
         pi_hidden: int,
         pi_nonlinearity: str,
         initial_steps: int):

    torch.set_num_threads(torch.get_num_threads())

    print(f'===== HYPERPARAMTERS =====')
    print(f'Domain and task: {domain} - {task}')
    print(f'Discount factor gamma: {gamma}')
    print(f'Target update weight tau: {tau}')
    print(f'Target update every: {target_update} steps')
    print(f'Learning rate: {learning_rate}')
    print(f'Update every {update_every} step')
    print(f'Number of episodes: {num_episodes}')
    print(f'Steps per episode: {steps_per_episode}')
    print(f'Random seed: {seed}')
    print(f'Temperature alpha: {temperature if temperature is not None else "automatic"}')
    print(f'Replay buffer size: {replay_size}')
    print(f'Replay buffer batch size: {batch_size}')
    print(f'Save intermediate: {save_intermediate}')
    print(f'Q hidden neurons: {q_hidden}')
    print(f'PI hidden neurons: {pi_hidden}')
    print(f'PI non-linearity: {pi_nonlinearity}')
    print(f'Number of tests to perform at every episode: {test_num}')
    print(f'Perform intial {initial_steps} steps in the environment to fill the buffer enough')

    # init seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setup logging utility
    writer = SummaryWriter(join(dirname(ROOT), 'experiments'))

    # init environment and query information
    env: Environment = suite.load(domain, task, task_kwargs={'random': seed})

    deterministic_test_env: Environment = suite.load(domain, task)
    random_test_evn: Environment = suite.load(domain, task)

    observation_spec = env.observation_spec()
    state_shape = np.sum([np.prod(value.shape, dtype=int) for value in observation_spec.values()])
    action_spec = env.action_spec()
    action_shape = np.prod(action_spec.shape, dtype=int)
    # assume all actions have the same maximum value
    max_action = action_spec.maximum[0]

    # initialize replay buffer
    replay_buffer = ReplayBuffer(size=replay_size, batch_size=batch_size)

    sac = SAC(state_shape, action_shape, max_action, gamma, tau, learning_rate, temperature, q_hidden, pi_hidden, pi_nonlinearity)

    updates = 0
    steps = num_episodes * steps_per_episode
    time_step = env.reset()
    rollout_return = 0
    rollout_length = 0
    updates = 0
    rollouts = 0

    for step in range(steps):
        # get current state
        state = flatten_and_concat(time_step.observation)

        # choose action for current step
        if step < initial_steps:
            action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
        else:
            with torch.no_grad():
                action, _ = sac.pi.act(to_tensor(state).unsqueeze(0))
            action = action.squeeze(0).numpy()

        # make a step with the environment
        time_step = env.step(action)
        next_state = flatten_and_concat(time_step.observation)
        reward = time_step.reward

        rollout_return += reward
        rollout_length += 1

        done = 0 if rollout_length == DM_ROLLOUT_LENGTH else time_step.last()

        replay_buffer.push(state, action, reward, next_state, done)

        if time_step.last():
            writer.add_scalar('return/train', rollout_return, rollouts)
            print(f"Rollout: {rollouts}, return: {round(rollout_return, 2)}")
            env.reset()
            rollout_length = 0
            rollout_return = 0
            rollouts += 1

        if replay_buffer.can_sample() and step % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample()
                q1_loss, q2_loss = sac.update_Q(batch)
                pi_loss = sac.update_pi(batch)
                sac.update_alpha(batch)
                if step % target_update == 0:
                    sac.update_targets()
                writer.add_scalar('loss/Q1', q1_loss, updates)
                writer.add_scalar('loss/Q2', q2_loss, updates)
                writer.add_scalar('loss/Q', q1_loss + q2_loss, updates)
                writer.add_scalar('loss/pi', pi_loss, updates)
                writer.add_scalar('alpha', sac.alpha.get(), updates)
                updates += 1

        # this is the last step of an episode (fixed length of steps)
        next_step = step + 1
        if next_step % steps_per_episode == 0:
            # test current policy
            deterministic_return = 0
            random_return = 0
            for _ in range(test_num):
                # test deterministicaly
                deterministic_step = deterministic_test_env.reset()
                while not deterministic_step.last():
                    state = flatten_and_concat(deterministic_step.observation)
                    with torch.no_grad():
                        action = sac.pi.act_deterministic(state)
                    deterministic_step = deterministic_test_env.step(action)
                    deterministic_return += deterministic_step.reward
                # test stochasticaly
                random_step = random_test_evn.reset()
                while not random_step.last():
                    state = flatten_and_concat(random_step.observation)
                    with torch.no_grad():
                        action, _ = sac.pi.act(to_tensor(state).unsqueeze(0))
                    random_step = random_test_evn.step(action.squeeze(0).numpy())
                    random_return += random_step.reward
            deterministic_avg_return = deterministic_return / test_num
            random_avg_return = random_return / test_num
            writer.add_scalar('deterministic_average_return/test', deterministic_avg_return, step // steps_per_episode)
            writer.add_scalar('random_average_return/test', random_avg_return, step // steps_per_episode)
            print(f"Deterministic test in episode: {step // steps_per_episode}, average return: {round(deterministic_avg_return, 2)}")
            print(f"Random test in episode: {step // steps_per_episode}, average return: {round(random_avg_return, 2)}")

            # save model just in case
            torch.save(sac.pi.state_dict(), f'trained_model_{step // steps_per_episode}.pt')

    env.close()
    torch.save(sac.pi.state_dict(), 'trained_model.pt')


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
