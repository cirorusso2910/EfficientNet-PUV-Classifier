from net.parameters.parameters_default import parameters_default

parameters_help = {

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    'mode': "EXECUTION MODE",
    'train': 'train model',
    'test': 'test model',
    'train_test': 'train and test model',

    'explainability': 'explainability mode',

    'who_is_my_creator': 'who is my creator?',

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    'dataset_path': f"dataset path (default: {parameters_default['dataset_path']})",
    'experiments_path': f"experiment path (default: {parameters_default['experiments_path']})",

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    'dataset': f"dataset name (default: '{parameters_default['dataset']}')",
    'image_height': f"image height (default: '{parameters_default['image_height']}')",
    'image_width': f"image width (default: '{parameters_default['image_width']}')",
    'split': f"dataset split (default: '{parameters_default['split']}')",
    'num_classes': f"number of classes (default: '{parameters_default['num_classes']}')",

    # ------------- #
    # EXPERIMENT ID #
    # ------------- #
    'typeID': f"experiment ID type (default: {parameters_default['typeID']}",
    'sep': f"separator in experiment ID (default: {parameters_default['sep']})",

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': f"GPU device name (default: {parameters_default['GPU']}",
    'num_threads': f"number of threads (default: {parameters_default['num_threads']}",

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': f"seed for reproducibility (default: {parameters_default['seed']})",

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    'norm': f"dataset normalization (default: {parameters_default['norm']}",

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    'resize': f"image resize shape (default: {parameters_default['resize']}",

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': f"batch size for train (default: {parameters_default['batch_size_train']})",
    'batch_size_val': f"batch size for validation (default: {parameters_default['batch_size_val']})",
    'batch_size_test': f"batch size for test (default: {parameters_default['batch_size_test']})",

    'num_workers': f"numbers of sub-processes to use for data loading, if 0 the data will be loaded in the main process (default: {parameters_default['num_workers']})",

    # ------- #
    # NETWORK #
    # ------- #
    'model': f"Network model (default: {parameters_default['model']})",
    'pretrained': f"PreTrained model (default: {parameters_default['pretrained']})",

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': f"number of epochs (default: {parameters_default['epochs']})",
    'epoch_to_resume': f"number of epoch to resume (default: {parameters_default['epoch_to_resume']})",

    'optimizer': f"Optimizer (default: '{parameters_default['optimizer']}'",
    'scheduler': f"Scheduler (default: '{parameters_default['scheduler']}'",
    'clip_gradient': f"Clip Gradient (default: '{parameters_default['clip_gradient']}'",

    'learning_rate': f"how fast approach the minimum (default: {parameters_default['learning_rate']})",
    'lr_momentum': f"momentum factor [optimizer: SGD] (default: {parameters_default['lr_momentum']})",
    'lr_patience': f"number of epochs with no improvement after which learning rate will be reduced [scheduler: ReduceLROnPlateau] (default: {parameters_default['lr_patience']})",
    'lr_step_size': f"how much the learning rate decreases [scheduler: StepLR] (default: {parameters_default['lr_step_size']})",
    'lr_gamma': f"multiplicative factor of learning rate decay [scheduler: StepLR] (default: {parameters_default['lr_gamma']})",
    'lr_T0': f"number of iterations until the first restart [scheduler: CosineAnnealingWarmRestarts] (default: {parameters_default['lr_T0']}]:",
    'lr_Tmax': f"maximum number of iterations [scheduler: CosineAnnealingLR] (default: {parameters_default['lr_Tmax']})",
    'max_norm': f"max norm of the gradients to be clipped [Clip Gradient] (default: {parameters_default['max_norm']})",

    # ---- #
    # LOSS #
    # ---- #
    'loss': f"loss function (default: {parameters_default['loss']})",
    'gamma': f"gamma parameter [loss: FocalLoss] (default: {parameters_default['gamma']}",

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    'load_best_accuracy_model': f"load best model with accuracy (default: {parameters_default['load_best_accuracy_model']})",

    # ------ #
    # OUTPUT #
    # ------ #
    'num_images': f"num images to show in test (default: {parameters_default['num_images']}",

}
