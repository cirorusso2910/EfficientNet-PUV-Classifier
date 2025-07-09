from typing import Union

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

from net.utility.msg.msg_save_model_complete import msg_save_best_model_complete
from net.utility.msg.msg_save_model_complete import msg_save_resume_model_complete


def save_best_model(epoch: int,
                    net: torch.nn.Module,
                    metrics: dict,
                    metrics_type: str,
                    optimizer: Union[Adam, SGD],
                    scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                    path: str):
    """
    Save best model

    :param epoch: num epoch
    :param net: net
    :param metrics: metrics
    :param metrics_type: metrics type
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        metrics_type: max(metrics),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save best-model
    msg_save_best_model_complete(metrics_type=metrics_type)


def save_resume_model(epoch: int,
                      net: torch.nn.Module,
                      accuracy: float,
                      MCC: float,
                      ROC_AUC: float,
                      PR_AUC: float,
                      optimizer: Union[Adam, SGD],
                      scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                      path: str):
    """
    Save resume model

    :param epoch: num epoch
    :param net: net
    :param accuracy: accuracy
    :param MCC: MCC
    :param ROC_AUC: ROC_AUC
    :param PR_AUC: PR_AUC
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save resume model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'accuracy': accuracy,
        'MCC': MCC,
        'ROC_AUC': ROC_AUC,
        'PR_AUC': PR_AUC,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)

    # msg save resume-model
    msg_save_resume_model_complete(epoch=epoch)
