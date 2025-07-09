import argparse
import sys
from typing import Union

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from net.utility.msg.msg_error import msg_error


def get_scheduler(optimizer: Union[Adam, SGD],
                  parser: argparse.Namespace) -> Union[ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts]:
    """
    Get scheduler

    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: scheduler
    """

    # ----------------- #
    # ReduceLROnPlateau #
    # ----------------- #
    if parser.scheduler == 'ReduceLROnPlateau' or parser.scheduler == 'ReduceLROP' or parser.scheduler == 'RLROP':
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      patience=parser.lr_patience)

    # ------ #
    # StepLR #
    # ------ #
    elif parser.scheduler == 'StepLR' or parser.scheduler == 'SLR':
        scheduler = StepLR(optimizer=optimizer,
                           step_size=parser.lr_step_size,
                           gamma=parser.lr_gamma)

    # --------------------------- #
    # CosineAnnealingWarmRestarts #
    # --------------------------- #
    elif parser.scheduler == 'CosineAnnealingWR' or parser.scheduler == 'CAWR':
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                T_0=parser.lr_T0)

    # --------------- #
    # CosineAnnealing #
    # --------------- #
    elif parser.scheduler == "CosineAnnealingLR" or parser.scheduler == 'CALR':
        scheduler = CosineAnnealingLR(optimizer=optimizer,
                                      T_max=parser.lr_Tmax)

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.scheduler,
                            type_variable='scheduler',
                            choices='[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR]')
        sys.exit(str_err)

    return scheduler
