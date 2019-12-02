import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
plt.style.use('seaborn-deep')

import types
from matplotlib.lines import Line2D

from solvers import *
from utils import Symbols as S

def _non_empty_subsquare_dim(array):
    nrows, ncols = array.shape
    for i in range(1, nrows):
        if (array[i:, :] == 0).all() and (array[:, i:] == 0).all():
            return i, i
    return nrows, ncols

def _plot_paper(array, ax, diff=None, attention=None, shape=(15, 15)):
    ncols, nrows = shape

    ax.set_xlim([0, ncols])
    ax.set_ylim([0, nrows])
    xticks = list(range(ncols))
    yticks = list(range(nrows))
    xticklabels = [2*' ' + l + ' ' for l in map(str, reversed(xticks))]
    yticklabels = [l for l in map(str, reversed(yticks))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha='left')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, va='bottom')

    ax.grid(True)
    if attention is None:
        attention = np.zeros((ncols, nrows))
        attention = np.random.randint(0, 2, (ncols, nrows))
    if diff is None:
        diff = np.zeros((ncols, nrows))
    for i in range(nrows):
        for j in range(ncols):
            symbol = array[i, j]
            fontdict = {
                'color': 'blue',
                'weight': 'bold',
                'fontsize': 'large',
            }
            if attention[i, j] > 0.5:
                fontdict['bbox'] = {
                                    'color': 'red',
                                    'alpha': 0.1
                                    }
            if diff[i, j]:
                fontdict['color'] = 'green'
            ax.text(j+0.5, ncols-(i+0.5), (S.visual(symbol)),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontdict=fontdict
                    )

def plot_last_step(steps):
    fig, ax = plt.subplots()
    _plot_paper(steps[-1], ax)
    fig.show()

def plot_steps(steps, ncols=None, title='Solution steps on paper',
               savefig=False):
    if ncols is None:
        ncols = 1 if len(steps) <= 2 else 2
    nrows = int(np.ceil(len(steps)/ncols))
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols, figsize=(ncols*6,nrows*6))
    if ncols == 1:
        axes = np.array(axes)
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    step_dims = [_non_empty_subsquare_dim(step.paper) for step in steps]
    nrows_paper = ncols_paper = np.max(step_dims)
    for i, step in enumerate(steps):
        step = steps[i]
        if i > 0:
            diff = (step['paper'] != steps[i-1]['paper'])
        else:
            diff = None
        axes[i].axis('on')
        axes[i].set_title(f"{i+1}. step:")
        _plot_paper(step['paper'], axes[i], diff=diff,
                    attention=step['attention'],
                    shape=(nrows_paper, ncols_paper))
    if savefig:
        fig.savefig(title + '.png', bbox_inches='tight')
    #plt.clf()
    #fig.show()

def _mock_problem_generator(problem):
    while True:
        yield problem

def plot_example(solver, problem, grid_size=100):
    res = next(iter(solver(grid_size).generator(
        _mock_problem_generator(problem))))
    plot_steps(res)
