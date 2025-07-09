import pandas

from sklearn.metrics import accuracy_score


def accuracy(classifications: pandas.DataFrame) -> float:
    """
    Compute accuracy

    :param classifications: classifications
    :return: accuracy score
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    accuracy_value = accuracy_score(y_true=true_labels,
                                    y_pred=predictions)

    return accuracy_value
