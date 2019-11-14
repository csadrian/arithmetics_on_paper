import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

symbols = {
    11: '+',
    12: '-',
    13: 'p?',
    14: '|',
    15: 'fact',
    16: '/',
    -1: ' '
}

for i in range(10):
    symbols[i] = str(i)

def _plot_paper(array, ax, diff=None, attention=None):
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
    fig, axes = plt.subplots(nrows=int(np.ceil(len(steps)/ncols)),
                             ncols=ncols, figsize=(8,8))
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
