from argparse import ArgumentParser, Namespace
from os.path import join
from os import makedirs
import matplotlib.pyplot as plt
import json
import numpy as np

Q_LOSS = {
    'loss_Q' : {
        'title' : 'Sum of losses of Q1 and Q2',
        'xlabel' : 'Steps',
        'ylabel' : 'Loss of Q1 + Q2',
        'format': 'sci',
    },
    'loss_Q1' : {
        'title' : 'Loss of Q1 state-action value function',
        'xlabel' : 'Steps',
        'ylabel' : 'Loss of Q1',
        'format': 'sci',
    },
    'loss_Q2' : {
        'title' : 'Loss of Q1 state-action value function',
        'xlabel' : 'Steps',
        'ylabel' : 'Loss of Q2',
        'format': 'sci',
    },
}

OTHER = {
    'loss_pi' : {
        'title' : 'Loss of policy π',
        'xlabel' : 'Steps',
        'ylabel' : 'Loss of π',
        'format': 'sci',
    },
    'alpha' : {
        'title' : 'Evolution of temperature parameter α',
        'xlabel' : 'Steps',
        'ylabel' : 'α',
        'format': 'sci',
    },
}

RETURN_PLOTS = {
    'return_train' : {
        'title' : 'Return of every training environment rollout',
        'xlabel' : 'Rollouts',
        'ylabel' : 'Return',
        'format': 'plain',
    },
}

TEST_PLOTS = {
    'title' : 'Test returns (10 run average)',
    'data' : [
        {
            'label': 'Deterministic policy π',
            'path': 'deterministic_average_return_test',
        },
        {
            'label': 'Stochastic policy π',
            'path': 'random_average_return_test',
        },
    ],
    'xlabel': 'Episodes',
    'ylabel': 'Average return',
    'format': 'plain',
}

BACKGROUND_COLOR = '#303030'
GRID_COLOR = '#525252'
FONT_COLOR = '#999999'
RED = '#ff7043'
BLUE = '#2196f3'

def parse_args() -> Namespace:
    """Parses arguments for evaluate()"""
    parser = ArgumentParser(
        description='Evaluates an agent on a Walker-Run environment.')
    parser.add_argument('--path', type=str)
    return parser.parse_args()

def load_data(path: str, fname: str):
    f = open(join(path, fname + '.json'), "r")

    data = json.load(f)
    steps = np.array([d[1] for d in data], dtype=np.int32)
    vals = np.array([d[2] for d in data], dtype=np.float32)

    f.close()

    return steps, vals

def prepare_ax(ax):
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.grid(visible=True, axis='both', color=GRID_COLOR)
    ax.set_box_aspect(1)
    ax.spines['bottom'].set_color(FONT_COLOR)
    ax.spines['top'].set_color(FONT_COLOR)
    ax.spines['left'].set_color(FONT_COLOR)
    ax.spines['right'].set_color(FONT_COLOR)
    ax.tick_params(axis='x', colors=FONT_COLOR)
    ax.tick_params(axis='y', colors=FONT_COLOR)


def plot_qs(path: str, outpath: str):
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, layout='tight', figsize=(15, 7))
    for ax, (name, data) in zip(axes.flat, Q_LOSS.items()):

        prepare_ax(ax)
        ax.set_title(data['title'], color=FONT_COLOR)
        ax.set_xlabel(data['xlabel'], color=FONT_COLOR)
        ax.set_ylabel(data['ylabel'], color=FONT_COLOR)
        ax.ticklabel_format(axis='both', style=data['format'], scilimits=(0, 0))

        steps, vals = load_data(path, name)
        ax.plot(steps, vals, color=RED)

    plt.tight_layout()
    fig.savefig(join(outpath, 'qloss.png'), transparent=False, facecolor=BACKGROUND_COLOR, dpi=500)

def plot_tests(path: str, outpath: str):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    prepare_ax(ax)
    ax.set_title(TEST_PLOTS['title'], color=FONT_COLOR)
    ax.set_xlabel(TEST_PLOTS['xlabel'], color=FONT_COLOR)
    ax.set_ylabel(TEST_PLOTS['ylabel'], color=FONT_COLOR)
    ax.ticklabel_format(axis='both', style=TEST_PLOTS['format'], scilimits=(0, 0))

    colors = [RED, BLUE]

    for data, color in zip(TEST_PLOTS['data'], colors):
        steps, vals = load_data(path, data['path'])
        ax.plot(steps, vals, color=color, label=data['label'])

    ax.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(join(outpath, 'tests.png'), transparent=False, facecolor=BACKGROUND_COLOR, dpi=500)

def plot_returns(path: str, outpath: str):
    for name, data in RETURN_PLOTS.items():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        prepare_ax(ax)
        ax.set_title(data['title'], color=FONT_COLOR)
        ax.set_xlabel(data['xlabel'], color=FONT_COLOR)
        ax.set_ylabel(data['ylabel'], color=FONT_COLOR)
        ax.ticklabel_format(axis='both', style=data['format'], scilimits=(0, 0))

        steps, vals = load_data(path, name)
        ax.plot(steps, vals, color=RED)

        plt.tight_layout()
        fig.savefig(join(outpath, name + '.png'), transparent=False, facecolor=BACKGROUND_COLOR, dpi=500)

def plot_others(path: str, outpath: str):
    for name, data in OTHER.items():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        prepare_ax(ax)
        ax.set_title(data['title'], color=FONT_COLOR)
        ax.set_xlabel(data['xlabel'], color=FONT_COLOR)
        ax.set_ylabel(data['ylabel'], color=FONT_COLOR)
        ax.ticklabel_format(axis='both', style=data['format'], scilimits=(0, 0))

        steps, vals = load_data(path, name)
        ax.plot(steps, vals, color=RED)

        plt.tight_layout()
        fig.savefig(join(outpath, name + '.png'), transparent=False, facecolor=BACKGROUND_COLOR, dpi=500)

def plot_all(path: str):
    outpath = join(path, 'plots')
    makedirs(outpath, exist_ok=True)

    plot_qs(path, outpath)
    plot_tests(path, outpath)
    plot_returns(path, outpath)
    plot_others(path, outpath)

if __name__ == "__main__":
    plot_all(**vars(parse_args()))
