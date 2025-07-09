import sys
from typing import List

from net.utility.msg.msg_error import msg_error


def metrics_header(metrics_type: str) -> List[str]:
    """
    Get metrics header according to type

    :param metrics_type: metrics type
    :return: header
    """

    if metrics_type == 'train':
        header = ["EPOCH",
                  "LOSS",
                  "LEARNING RATE",
                  "ACCURACY",
                  "MCC",
                  "ROC AUC",
                  "PR AUC",
                  "TIME TRAIN",
                  "TIME VALIDATION",
                  "TIME METRICS"]

    elif metrics_type == 'test':
        header = ["ACCURACY",
                  "MCC",
                  "ROC AUC",
                  "PR AUC",
                  "TIME TEST",
                  "TIME METRICS"]

    elif metrics_type == 'test_class':
        header = ["CLASS",
                  "PRECISION",
                  "RECALL",
                  "F1-SCORE"]

    elif metrics_type == 'test_PR':
        header = ["AVERAGE",
                  "PRECISION",
                  "RECALL",
                  "F1-SCORE",]

    else:
        str_err = msg_error(file=__file__,
                            variable=metrics_type,
                            type_variable='header metrics type',
                            choices='[train, test, test_class, test_PR]')
        sys.exit(str_err)

    return header
