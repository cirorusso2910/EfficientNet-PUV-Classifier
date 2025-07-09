parameters_default = {

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    'dataset_path': 'datasets',
    'experiments_path': 'experiments',

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    'dataset': 'VCUG',
    'image_height': 1120,
    'image_width': 576,
    'split': 'default',
    'num_classes': 2,

    # ------------- #
    # EXPERIMENT ID #
    # ------------- #
    'typeID': 'default',
    'sep': '|',

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': 'None',
    'num_threads': 32,

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': 0,

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    'norm': 'none',

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    'resize': 224,

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': 8,
    'batch_size_val': 8,
    'batch_size_test': 8,

    'num_workers': 8,

    # ------- #
    # NETWORK #
    # ------- #
    'model': 'ResNet-18',
    'pretrained': False,

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': 1,
    'epoch_to_resume': 0,

    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'clip_gradient': True,

    'learning_rate': 1e-4,
    'lr_momentum': 0.1,  # optimizer: SGD
    'lr_patience': 3,  # scheduler: ReduceLROnPlateau
    'lr_step_size': 3,  # scheduler: StepLR
    'lr_gamma': 0.1,  # scheduler: StepLR
    'lr_T0': 10,  # scheduler: CosineAnnealingWarmRestarts
    'lr_Tmax': 25,  # scheduler: CosineAnnealingLR
    'max_norm': 0.1,

    # ---- #
    # LOSS #
    # ---- #
    'loss': 'CrossEntropyLoss',
    'gamma': 2.0,  # loss: FocalLoss

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    'load_best_accuracy_model': True,

    # ------ #
    # OUTPUT #
    # ------ #
    'num_images': 1,

}
