import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Compute_accuracy(U, U_true, thresh=0):

    U = np.vectorize(round)(U)
    return (np.abs(U - U_true) <= thresh).sum()/(np.prod(np.shape(U)))


def plot_similarity(U, title='', save_path='/media/sf_Debian-shared-folder/', name='fig'):

    U = np.vectorize(round)(U)
    sns.set(style="white")
    mask = np.zeros_like(U, dtype=np.bool)
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(U, mask=mask, cmap=cmap, vmax=U.max(), center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.savefig('{0}/{1}.png'.format(save_path, name))


def plot_n_clusters(n_clusters, True_n_clusters, title='', save_path='/media/sf_Debian-shared-folder/', name='fig'):

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(np.linspace(0, len(n_clusters), num=len(n_clusters)), n_clusters)
    if True_n_clusters is not None :
        ax.axhline(y=True_n_clusters, color='r', linestyle='-')
    plt.title(title)
    plt.savefig('{0}/{1}.png'.format(save_path, name))

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    '''
    Credit to tonysyu.github.io
    '''
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

