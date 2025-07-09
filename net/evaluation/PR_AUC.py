import numpy as np
import pandas

from sklearn.metrics import average_precision_score


def PR_AUC(classifications: pandas.DataFrame) -> float:
    """
    Compute the Area Under the PR curve

    :param classifications: classifications
    :return: PR AUC value
    """

    predictions = classifications['PREDICTION']  # predictions
    true_labels = classifications['LABEL']  # true labels (ground truth)
    label_bin = np.where(predictions == true_labels, 1, 0)  # label binary

    scores = classifications['SCORE'].values  # scores

    try:
        PR_AUC = average_precision_score(y_true=label_bin,
                                         y_score=scores)
    except ValueError:
        PR_AUC = 0

    return PR_AUC
