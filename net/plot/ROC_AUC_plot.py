from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from net.metrics.utility.best_metrics import best_metrics
from net.utility.msg.msg_plot_complete import msg_plot_complete


def ROC_AUC_plot(figsize: Tuple[int, int],
                 title: str,
                 experiment_ID: str,
                 ticks: List[int],
                 epochs_ticks: np.ndarray,
                 ROC_AUC: List[float],
                 accuracy_path: str):
    """
    ROC AUC plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param ROC_AUC: ROC_AUC
    :param accuracy_path: accuracy path
    """

    # best metrics
    max_ROC_AUC, index_max_ROC_AUC = best_metrics(metric=ROC_AUC)

    # Figure: ROC AUC
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, ROC_AUC, marker=".", color='blue', label='ROC AUC')
    plt.scatter(x=index_max_ROC_AUC, y=max_ROC_AUC, marker="x", color='blue', label='Best ROC AUC')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("ROC AUC")
    plt.ylim(0.0, 1.0)
    plt.savefig(accuracy_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)

    # plot complete
    msg_plot_complete(plot_type='ROC AUC')
