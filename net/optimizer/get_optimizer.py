import argparse
import sys
from typing import Union, Iterator

from torch.nn import Parameter
from torch.optim import Adam, SGD, AdamW

from net.utility.msg.msg_error import msg_error


def get_optimizer(net_parameters: Iterator[Parameter],
                  parser: argparse.Namespace) -> Union[Adam, SGD]:
    """
    Get optimizer

    :param net_parameters: net parameters
    :param parser: parser of parameters-parsing
    :return: optimizer
    """

    # ---- #
    # ADAM #
    # ---- #
    if parser.optimizer == 'Adam' or parser.optimizer == 'A':
        optimizer = Adam(params=net_parameters,
                         lr=parser.learning_rate)

    # ------ #
    # ADAM W #
    # ------ #
    elif parser.optimizer == 'AdamW' or parser.optimizer == 'AW':
        optimizer = AdamW(params=net_parameters,
                          lr=parser.learning_rate)

    # --- #
    # SGD #
    # --- #
    elif parser.optimizer == 'SGD':
        optimizer = SGD(params=net_parameters,
                        lr=parser.learning_rate,
                        momentum=parser.lr_momentum)

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.optimizer,
                            type_variable='optimizer',
                            choices='[Adam, AdamW, SGD]')
        sys.exit(str_err)

    return optimizer
