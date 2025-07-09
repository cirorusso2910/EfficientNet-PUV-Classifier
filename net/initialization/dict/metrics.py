import sys

from net.utility.msg.msg_error import msg_error


def metrics_dict(metrics_type: str) -> dict:
    """
    Get metrics dictionary according to type

    :param metrics_type: metrics type
    :return: metrics dictionary
    """

    if metrics_type == 'train':
        metrics = {
            'ticks': [],

            'loss': [],
            'learning_rate': [],

            'accuracy': [],
            'MCC': [],
            'ROC_AUC': [],
            'PR_AUC': [],

            'time': {
                'train': [],
                'validation': [],
                'metrics': []
            }
        }

    elif metrics_type == 'test':
        metrics = {
            'accuracy': [],
            'MCC': [],
            'ROC_AUC': [],
            'PR_AUC': [],

            'precision': {
                'micro': [],
                'macro': [],
                'weighted': []
            },

            'recall': {
                'micro': [],
                'macro': [],
                'weighted': []
            },

            'F1': {
                'micro': [],
                'macro': [],
                'weighted': []
            },

            'class': {
                'precision': [],
                'recall': [],
                'F1': [],
            },

            'time': {
                'test': [],
                'metrics': []
            }
        }

    else:
        str_err = msg_error(file=__file__,
                            variable=metrics_type,
                            type_variable='metrics type',
                            choices='[train, test]')
        sys.exit(str_err)

    return metrics
