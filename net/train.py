import argparse
import sys
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader

from net.loss.classes.MyFocalLoss import MyFocalLoss
from net.metrics.utility.timer import timer
from net.utility.msg.msg_error import msg_error


def train(num_epoch: int,
          epochs: int,
          net: torch.nn.Module,
          dataloader: DataLoader,
          optimizer: Union[SGD],
          scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR],
          loss_name: str,
          criterion: Union[CrossEntropyLoss, MyFocalLoss],
          device: torch.device,
          parser: argparse.Namespace) -> float:
    """
    Training function

    :param num_epoch: number of epochs
    :param epochs: epochs
    :param net: net
    :param dataloader: dataloader
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param loss_name: loss name
    :param criterion: criterion (loss)
    :param device: device
    :param parser: parser of parameters-parsing
    :return: average epoch loss
    """

    # switch to train mode
    net.train()

    # reset performance measures
    epoch_loss_hist = []

    # for each batch in dataloader
    for num_batch, batch in enumerate(dataloader):

        # init batch time
        time_batch_start = time.time()

        # get data from dataloader
        filename, image, label = batch['filename'], batch['image'].float().to(device), batch['label'].to(device)

        # zero (init) the parameter gradients
        optimizer.zero_grad()

        # forward pass
        classifications = net(image)

        # calculate loss
        if loss_name == 'CrossEntropyLoss' or loss_name == 'CE':
            loss = criterion(classifications, label)
        elif loss_name == 'WeightedCrossEntropyLoss' or loss_name == 'WCE':
            loss = criterion(classifications, label)
        elif loss_name == 'BCELoss' or loss_name == 'BCE':
            # transforms label [B] in [B x 2] (2 class)
            label_one_hot = F.one_hot(label, num_classes=2).float()
            loss = criterion(classifications, label_one_hot)
        elif loss_name == 'FocalLoss' or loss_name == 'FL':
            loss = criterion(classifications, label)
        else:
            str_err = msg_error(file=__file__,
                                variable=loss_name,
                                type_variable='loss',
                                choices='[CrossEntropyLoss (CE), WeightedCrossEntropyLoss (WCE), BCELoss (BCE), FocalLoss (FL)]')
            sys.exit(str_err)

        # append epoch loss
        epoch_loss_hist.append(float(loss.item()))

        # loss gradient backpropagation
        loss.backward()

        # clip gradient
        if parser.clip_gradient:
            clip_grad_norm_(parameters=net.parameters(),
                            max_norm=parser.max_norm)

        # net parameters update
        optimizer.step()

        # batch time
        time_batch = time.time() - time_batch_start

        # batch time conversion
        batch_time = timer(time_elapsed=time_batch)

        print("Epoch: {}/{} |".format(num_epoch, epochs),
              "Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
              "Loss: {:1.5f} |".format(float(loss)),
              "Time: {:.0f} s ".format(batch_time['seconds']))

        del loss

    # step learning rate scheduler
    if parser.scheduler == 'ReduceLROnPlateau':
        scheduler.step(np.mean(epoch_loss_hist))
    elif parser.scheduler == 'StepLR':
        scheduler.step()
    elif parser.scheduler == 'CosineAnnealingWR':
        scheduler.step()
    elif parser.scheduler == 'CosineAnnealingLR':
        scheduler.step()

    return sum(epoch_loss_hist) / len(dataloader)
