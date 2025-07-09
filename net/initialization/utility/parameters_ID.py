import argparse

from net.device.get_GPU_name import get_GPU_name
from net.metrics.utility.my_notation import scientific_notation


def parameters_ID(parser: argparse.Namespace) -> dict:
    """
    Get parameters ID

    :param parser: parser of parameters-parsing
    :return: parameters ID dictionary
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    dataset_ID = "dataset={}".format(parser.dataset)
    norm_ID = "norm={}".format(parser.norm)
    split_ID = "split={}".format(parser.split)
    num_classes_ID = "classes={}".format(parser.num_classes)
    resize_ID = "resize={}".format(parser.resize)
    ep_ID = "ep={}".format(parser.epochs)
    loss_ID = "loss={}".format(parser.loss)
    optimizer_ID = "optimizer={}".format(parser.optimizer)
    scheduler_ID = "scheduler={}".format(parser.scheduler)
    clip_gradient_ID = "clip_gradient={}".format(parser.clip_gradient)
    lr_ID = "lr={}".format(scientific_notation(parser.learning_rate))
    lr_momentum_ID = "lr_momentum={}".format(parser.lr_momentum)
    lr_patience_ID = "lr_patience={}".format(parser.lr_patience)
    lr_step_size_ID = "lr_step_size={}".format(parser.lr_step_size)
    lr_gamma_ID = "lr_gamma={}".format(parser.lr_gamma)
    lr_T0_ID = "lr_T0={}".format(parser.lr_T0)
    lr_Tmax_ID = "lr_Tmax={}".format(parser.lr_Tmax)
    bs_ID = "bs={}".format(parser.batch_size_train)
    model_ID = "model={}".format(parser.model)
    pretrained_ID = "pretrained={}".format(parser.pretrained)
    if 'script' in parser.mode:
        GPU_ID = "GPU={}".format(parser.GPU)
    else:
        GPU_ID = "GPU={}".format(get_GPU_name())

    # customized scheduler
    if parser.scheduler == 'ReduceLROnPlateau':
        scheduler_customized_ID = scheduler_ID + "-{}".format(parser.lr_patience)
    elif parser.scheduler == 'StepLR':
        scheduler_customized_ID = scheduler_ID + "-{}".format(parser.lr_step_size)
    elif parser.scheduler == 'CosineAnnealingWR':
        scheduler_customized_ID = scheduler_ID + "-{}".format(parser.lr_T0)
    elif parser.scheduler == 'CosineAnnealingLR':
        scheduler_customized_ID = scheduler_ID + "-{}".format(parser.lr_Tmax)
    else:
        scheduler_customized_ID = scheduler_ID

    # customized loss
    if parser.loss == 'CrossEntropyLoss' or parser.loss == 'CE':
        loss_customized_ID = "loss=CE"
    elif parser.loss == 'WeightedCrossEntropyLoss' or parser.loss == 'WCE':
        loss_customized_ID = "loss=WCE"
    elif parser.loss == 'BCELoss' or parser.loss == 'BCE':
        loss_customized_ID = "loss=BCE"
    elif parser.loss == 'FocalLoss' or parser.loss == 'FL':
        loss_customized_ID = "loss=FL"
    else:
        loss_customized_ID = loss_ID

    parameters_ID_dict = {
        'dataset': dataset_ID,
        'norm': norm_ID,
        'split': split_ID,
        'num_classes': num_classes_ID,
        'resize': resize_ID,
        'ep': ep_ID,
        'loss': loss_ID,
        'loss_custom': loss_customized_ID,
        'optimizer': optimizer_ID,
        'scheduler': scheduler_ID,
        'scheduler_custom': scheduler_customized_ID,
        'clip_gradient': clip_gradient_ID,
        'lr': lr_ID,
        'lr_momentum': lr_momentum_ID,
        'lr_patience': lr_patience_ID,
        'lr_step_size': lr_step_size_ID,
        'lr_gamma': lr_gamma_ID,
        'lr_T0': lr_T0_ID,
        'lr_Tmax': lr_Tmax_ID,
        'bs': bs_ID,
        'model': model_ID,
        'pretrained': pretrained_ID,
        'GPU': GPU_ID
    }

    return parameters_ID_dict
