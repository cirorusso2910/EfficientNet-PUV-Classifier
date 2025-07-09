import argparse
import sys
from typing import Union

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from net.utility.msg.msg_error import msg_error


def current_learning_rate(scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts],
                          optimizer: Union[Adam, AdamW, SGD],
                          parser: argparse.Namespace) -> float:
    """
    Get current learning rate according to scheduler and optimizer type

    :param scheduler: scheduler
    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: current learning rate
    """

    if parser.scheduler == 'ReduceLROnPlateau' or parser.scheduler == 'ReduceLROP' or parser.scheduler == 'RLROP':
        learning_rate = my_get_last_lr(optimizer=optimizer)

    elif parser.scheduler == 'StepLR' or parser.scheduler == 'SLR':
        learning_rate = scheduler.get_last_lr()[0]

    elif parser.scheduler == 'CosineAnnealingLR' or parser.scheduler == 'CALR':
        learning_rate = scheduler.get_last_lr()[0]

    elif parser.scheduler == 'CosineAnnealingWR' or parser.scheduler == 'CAWR':
        learning_rate = scheduler.get_last_lr()[0]

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.scheduler,
                            type_variable='scheduler',
                            choices='[ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWR]')
        sys.exit(str_err)

    return learning_rate


def my_get_last_lr(optimizer: Union[Adam, AdamW, SGD]) -> float:
    """
    Get last Learning Rate
    :param optimizer: optimizer
    :return: last learning rate
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']
