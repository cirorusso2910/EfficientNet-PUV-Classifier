import numpy as np
import pandas
from sklearn.metrics import confusion_matrix


def my_confusion_matrix(classifications: pandas.DataFrame) -> np.array:
    """
    Compute confusion matrix

    :param classifications: classifications
    :return: confusion matrix
    """

    ground_truth = classifications['LABEL'].values  # ground truth
    prediction = classifications['PREDICTION'].values  # prediction

    # compute confusion matrix
    confusion_matrix_result = confusion_matrix(y_true=ground_truth,
                                               y_pred=prediction)

    return confusion_matrix_result
