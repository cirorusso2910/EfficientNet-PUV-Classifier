import argparse

from net.dataset.utility.get_image_shape import get_image_shape
from net.metrics.utility.my_notation import scientific_notation


def parameters_summary(parser: argparse.Namespace,
                       num_images: dict,
                       num_label: dict):
    """
    Show parameters summary

    :param parser: parser of parameters-parsing
    :param num_images: number images dictionary
    :param num_label: num label dictionary
    """

    print("\n-------------------"
          "\nPARAMETERS SUMMARY:"
          "\n-------------------")

    # image shape
    image_height, image_width = get_image_shape()

    # ------- #
    # DATASET #
    # ------- #
    print("\nDATASET:"
          "\nDataset name: {}".format(parser.dataset),
          "\nImage Shape (H x W): {} x {}".format(image_height, image_width),
          "\nData Split: {}".format(parser.split))

    # ------------- #
    # DATASET SPLIT #
    # ------------- #
    print("\nDATASET SPLIT")
    for split in ['train', 'validation', 'test']:
        print("{} data: {} images".format(split.capitalize(), num_images[split]))
        label_stats = num_label[split]
        label_info = " | ".join("{}: {}".format(label, count) for label, count in label_stats.items())
        print(label_info)

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    print("\nDATASET TRANSFORMS:"
          "\nResize: {} x {}".format(parser.resize, parser.resize))

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    print("\nDATASET NORMALIZATION:"
          "\nNormalization: {}".format(parser.norm))

    # ---------- #
    # DATALOADER #
    # ---------- #
    print("\nDATALOADER:"
          "\nBatch size train: {}".format(parser.batch_size_train),
          "\nBatch size validation: {}".format(parser.batch_size_val),
          "\nBatch size test: {}".format(parser.batch_size_test))

    # ------------- #
    # NETWORK MODEL #
    # ------------- #
    print("\nNETWORK MODEL:"
          "\nModel: {}".format(parser.model),
          "\nPreTrained: {}".format(parser.pretrained))

    # ---------------- #
    # HYPER PARAMETERS #
    # ---------------- #
    print("\nHYPER PARAMETERS:"
          "\nEpochs: {}".format(parser.epochs),
          "\nOptimizer: {}".format(parser.optimizer),
          "\nScheduler: {}".format(parser.scheduler),
          "\nClip Gradient: {}".format(parser.clip_gradient),
          "\nLearning Rate: {}".format(scientific_notation(parser.learning_rate)))

    if parser.optimizer == 'SGD':
        print("Momentum: {}".format(parser.lr_momentum))

    if parser.scheduler == 'ReduceLROnPlateau':
        print("Patience: {}".format(parser.lr_patience))

    elif parser.scheduler == 'StepLR':
        print("Step Size: {}".format(parser.lr_step_size))

    elif parser.scheduler == 'CosineAnnealingWR':
        print("T0: {}".format(parser.lr_T0))

    elif parser.scheduler == 'CosineAnnealingLR':
        print("T_Max: {}".format(parser.lr_Tmax))

    if parser.clip_gradient:
        print("Max Norm: {}".format(parser.max_norm))

    # --------- #
    # CRITERION #
    # --------- #
    print("\nCRITERION:"
          "\nLoss: {}".format(parser.loss))

    if parser.loss == 'SigmoidFocalLoss':
        print("Alpha: {}".format(parser.alpha),
              "\nGamma: {}".format(parser.gamma))

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    if parser.mode in ['test']:
        print("\nLOAD MODEL:")
        if parser.load_best_accuracy_model:
            print("Load best accuracy model")
