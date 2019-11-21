from solvers import *
import solvers
from solvers_milan import *
import solvers_milan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

symbols = {
    solvers.SIGN_ADD: '+',
    solvers.SIGN_SUB: '-',
    solvers.SIGN_IS_PRIME: 'p?',
    solvers.SIGN_IS_DIVISIBLE_BY: '|',
    solvers.SIGN_FACTORIZE: 'fact',
    solvers.SIGN_DIV: '/',
    -1: ' ',
    solvers.SIGN_YES: '✓',
    solvers.SIGN_NO: 'X',
    solvers.SIGN_SQRT: '√',
    solvers.SIGN_BASE_CONVERSION: '→b',
    solvers.SIGN_PRODUCT: '*',
    solvers.SIGN_EQ: '=',
}

for i in range(10):
    symbols[i] = str(i)

def _plot_paper(array, ax, diff=None, attention=None, transform=True):
    if transform:
        array = np.rot90(array, 3)
        if diff is not None:
            diff = np.rot90(diff, 3)
        if attention is not None:
            attention = np.rot90(attention, 3)
    nrows, ncols = array.shape
    ax.set_xlim([0, ncols])
    ax.set_ylim([0, nrows])
    ax.set_xticks(range(ncols+1))
    ax.set_yticks(range(nrows+1))
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
                fontdict['bbox'] = {'facecolor': 'white',
                                    'color': 'red',
                                    'alpha': 0.1
                                    }
            if diff[i, j]:
                fontdict['color'] = 'green'
            ax.text(i+0.5, j+0.5, (symbols[symbol]),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontdict=fontdict 
                    )

def plot_last_step(steps):
    fig, ax = plt.subplots()
    _plot_paper(steps[-1], ax)
    fig.show()

def plot_steps(steps, ncols=None):
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
    for i, step in enumerate(steps):
        step = steps[i]
        if i > 0:
            diff = (step['paper'] != steps[i-1]['paper'])
        else:
            diff = None
        axes[i].axis('on')
        axes[i].set_title(f"{i+1}. step:")
        _plot_paper(step['paper'], axes[i], diff=diff,
                    attention=step['attention'])
    fig.show()
