from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from net.metrics.utility.best_metrics import best_metrics
from net.utility.msg.msg_plot_complete import msg_plot_complete


def MCC_plot(figsize: Tuple[int, int],
             title: str,
             experiment_ID: str,
             ticks: List[int],
             epochs_ticks: np.ndarray,
             MCC: List[float],
             MCC_path: str):
    """
    MCC plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param MCC: MCC
    :param MCC_path: MCC path
    """

    # best metrics
    max_MCC, index_MCC = best_metrics(metric=MCC)

    # Figure: MCC
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, MCC, marker=".", color='blue', label='MCC')
    plt.scatter(x=index_MCC, y=max_MCC, marker="x", color='blue', label='Best MCC')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("MCC")
    plt.ylim(-1.0, 1.0)
    plt.savefig(MCC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='MCC')
