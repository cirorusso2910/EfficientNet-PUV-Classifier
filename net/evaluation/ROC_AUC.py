import numpy as np
import pandas
from sklearn.metrics import roc_auc_score


def ROC_AUC(classifications: pandas.DataFrame) -> float:
    """
    Compute the Area Under the ROC curve

    :param classifications: classifications
    :return: accuracy score
    """

    predictions = classifications['PREDICTION']  # predictions
    true_labels = classifications['LABEL']  # true labels (ground truth)
    label_bin = np.where(predictions == true_labels, 1, 0)  # label binary

    scores = classifications['SCORE'].values  # scores

    try:
        ROC_AUC_value = roc_auc_score(y_true=label_bin,
                                      y_score=scores)
    except ValueError:
        ROC_AUC_value = 0

    return ROC_AUC_value
