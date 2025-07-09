import numpy as np
import pandas
from pandas import DataFrame

from net.dataset.utility.get_class_name import get_class_name


def confusion_matrix_test_csv(classifications: pandas.DataFrame,
                              confusion_matrix: np.array,
                              confusion_matrix_path: str,
                              label_path: str
                              ):
    """
    Save confusion-matrix-test-classification.csv

    :param classifications: classifications
    :param confusion_matrix: confusion_matrix
    :param confusion_matrix_path: confusion matrix path
    :param label_path: label path
    """

    # class label
    class_label = np.unique(classifications["LABEL"])

    # build class labels
    class_labels = [get_class_name(label_path=label_path, label_to_find=idx_label) for idx_label in class_label]

    # create DataFrame
    confusion_matrix_df = DataFrame(confusion_matrix, columns=class_labels, index=class_labels)

    # save confusion-matrix.csv
    confusion_matrix_df.to_csv(confusion_matrix_path)
