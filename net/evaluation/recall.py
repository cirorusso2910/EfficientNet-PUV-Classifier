from typing import List, Any

import pandas

from sklearn.metrics import recall_score


def recall_per_class(classifications: pandas.DataFrame) -> List[float]:
    """
    Compute recall per class

    :param classifications: classifications
    :return: recall
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    recall_value = recall_score(y_true=true_labels,
                                y_pred=predictions,
                                average=None)  # per class

    return recall_value.tolist()


def recall(classifications: pandas.DataFrame,
           average: Any) -> float:
    """
    Compute recall

    :param classifications: classifications
    :param average: average
    :return: recall
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    recall_value = recall_score(y_true=true_labels,
                                y_pred=predictions,
                                average=average)

    return recall_value
