"""Evaluates an agent on a DMControl environment.

DO NOT MODIFY THIS FILE!!!
"""
from argparse import ArgumentParser, Namespace
from math import fsum
from pickle import dump
from typing import List, Optional

from dm_control import suite
from dm_control import viewer
from dm_control.rl.control import Environment

from ee619.agent import Agent


def parse_args() -> Namespace:
    """Parses arguments for evaluate()"""
    parser = ArgumentParser(
        description='Evaluates an agent on a Walker-Run environment.')
    parser.add_argument('-l', dest='label', default=None,
                        help='if unspecified, the mean episodic return will be'
                             ' printed to stdout. otherwise, it will be dumped'
                             ' to a pickle file of the given path.')
    parser.add_argument('-n', type=int, dest='repeat', default=1,
                        help='number of trials.')
    parser.add_argument('-s', type=int, dest='seed', default=0,
                        help='passed to the environment for determinism.')
    parser.add_argument('--domain', default='walker',
                        help='DMControl domain to evaluate the agent')
    parser.add_argument('--task', default='run',
                        help='DMControl task of interest')
    return parser.parse_args()


if __name__ == '__main__':
        env: Environment = suite.load(domain_name='walker',
                                      task_name='run', task_kwargs={"random": 42})
        agent = Agent()
        agent.load()
        viewer.launch(env, policy=(lambda time_step: agent.act(time_step)), height=1080, width=1920)
