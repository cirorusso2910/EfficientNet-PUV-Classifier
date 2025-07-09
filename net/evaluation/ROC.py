from typing import Tuple

import numpy as np
import pandas
from sklearn.metrics import roc_curve


def ROC(classifications: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC Curve

    :param classifications: classifications
    :return: False Positive Rate (FPR) and True Positive Rate (TPR)
    """

    predictions = classifications['PREDICTION']  # predictions
    true_labels = classifications['LABEL']  # true labels (ground truth)
    label_bin = np.where(predictions == true_labels, 1, 0)  # label binary

    scores = classifications['SCORE'].values  # scores

    # ROC curve
    FPR, TPR, thresholds = roc_curve(y_true=label_bin, y_score=scores, pos_label=1)

    return FPR, TPR
