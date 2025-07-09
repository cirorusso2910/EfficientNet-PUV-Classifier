from typing import List, Any

import pandas

from sklearn.metrics import precision_score


def precision_per_class(classifications: pandas.DataFrame) -> List[float]:
    """
    Compute precision per class

    :param classifications: classifications
    :return: precision
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    precision_value = precision_score(y_true=true_labels,
                                      y_pred=predictions,
                                      average=None)  # per class

    return precision_value.tolist()


def precision(classifications: pandas.DataFrame,
              average: Any) -> float:
    """
    Compute precision

    :param classifications: classifications
    :param average: average
    :return: precision
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    precision_value = precision_score(y_true=true_labels,
                                      y_pred=predictions,
                                      average=average)

    return precision_value
