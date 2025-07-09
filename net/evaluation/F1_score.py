from typing import List, Any

import pandas

from sklearn.metrics import f1_score


def F1_score_per_class(classifications: pandas.DataFrame) -> List[float]:
    """
    Compute F1-Score per class

    :param classifications: classifications
    :return: F1-Score
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    f1_score_value = f1_score(y_true=true_labels,
                              y_pred=predictions,
                              average=None)  # per class

    return f1_score_value.tolist()


def F1_score(classifications: pandas.DataFrame,
             average: Any) -> float:
    """
    Compute F1-Score

    :param classifications: classifications
    :return: F1-Score
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    f1_score_value = f1_score(y_true=true_labels,
                              y_pred=predictions,
                              average=average)

    return f1_score_value
