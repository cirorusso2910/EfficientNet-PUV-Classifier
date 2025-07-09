import pandas
from sklearn.metrics import matthews_corrcoef


def MCC(classifications: pandas.DataFrame) -> float:
    """
    Compute the Matthews correlation coefficient (MCC)

    :param classifications: classifications
    :return: MCC value
    """

    predictions = classifications['PREDICTION'].values  # predictions
    true_labels = classifications['LABEL'].values  # true labels (ground truth)

    # MCC
    MCC_value = matthews_corrcoef(y_true=true_labels,
                                  y_pred=predictions)

    return MCC_value
