from typing import Tuple

import numpy as np
import pandas
from sklearn.metrics import precision_recall_curve


def PR(classifications: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall Curve

    :param classifications: classifications
    :return: precision and recall
    """

    predictions = classifications['PREDICTION']  # predictions
    true_labels = classifications['LABEL']  # true labels (ground truth)
    label_bin = np.where(predictions == true_labels, 1, 0)  # label binary

    scores = classifications['SCORE'].values  # scores

    # compute the Precision-Recall Curve (PR)
    precision, recall, thresholds = precision_recall_curve(label_bin, scores)

    return precision, recall
