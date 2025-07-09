import argparse
import sys
from typing import Union

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from net.loss.classes.MyFocalLoss import MyFocalLoss, compute_alpha
from net.loss.utility.compute_weights import compute_weights
from net.utility.msg.msg_error import msg_error


def get_loss(loss_name: str,
             num_label: dict,
             parser: argparse.Namespace,
             device: torch.device) -> Union[CrossEntropyLoss, MyFocalLoss]:
    """
    Get loss

    :param parser: parser of parameters-parsing
    :param num_label: num label (all dataset)
    :return: criterion (loss)
    """

    # ------------------ #
    # CROSS ENTROPY LOSS #
    # ------------------ #
    if loss_name == 'CrossEntropyLoss' or loss_name == 'CE':
        criterion = CrossEntropyLoss()

    # --------------------------- #
    # WEIGHTED CROSS ENTROPY LOSS #
    # --------------------------- #
    elif loss_name == 'WeightedCrossEntropyLoss' or loss_name == 'WCE':

        # compute weights
        weights = compute_weights(num_label=num_label).to(device)

        criterion = CrossEntropyLoss(weight=weights)

    # ------------------------- #
    # BINARY CROSS ENTROPY LOSS #
    # ------------------------- #
    elif loss_name == 'BCELoss' or loss_name == 'BCE':

        # compute weights
        weights = compute_weights(num_label=num_label).to(device)

        criterion = BCEWithLogitsLoss(weight=weights)

    # ---------- #
    # FOCAL LOSS #
    # ---------- #
    elif loss_name == 'FocalLoss' or loss_name == 'FL':

        # compute alpha
        alpha = compute_alpha(num_label=num_label).to(device)

        criterion = MyFocalLoss(alpha=alpha,
                                gamma=parser.gamma)

    else:
        str_err = msg_error(file=__file__,
                            variable=loss_name,
                            type_variable='loss',
                            choices='[CrossEntropyLoss, WeightedCrossEntropyLoss, BCELoss, FocalLoss]')
        sys.exit(str_err)

    return criterion
