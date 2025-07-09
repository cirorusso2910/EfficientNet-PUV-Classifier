import argparse

from net.parameters.parameters_choices import parameters_choices
from net.parameters.parameters_default import parameters_default
from net.parameters.parameters_help import parameters_help


def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    parser_mode = parser.add_subparsers(title=parameters_help['mode'], dest='mode', metavar='mode')

    # execution mode
    parser_train = parser_mode.add_parser('train', help=parameters_help['train'])
    parser_test = parser_mode.add_parser('test', help=parameters_help['test'])
    parser_train_test = parser_mode.add_parser('train_test', help=parameters_help['train_test'])

    parser_explainability = parser_mode.add_parser('explainability', help=parameters_help['explainability'])

    # who is my creator
    parser_who_is_my_creator = parser_mode.add_parser('who_is_my_creator', help=parameters_help['who_is_my_creator'])

    # execution mode list
    execution_mode = [parser_train,
                      parser_test,
                      parser_train_test,
                      parser_explainability,
                      parser_who_is_my_creator]

    # for each subparser 'mode'
    for subparser in execution_mode:

        # -------------- #
        # INITIALIZATION #
        # -------------- #
        subparser.add_argument('--dataset_path',
                               type=str,
                               default=parameters_default['dataset_path'],
                               help=parameters_help['dataset_path'])

        subparser.add_argument('--experiments_path',
                               type=str,
                               default=parameters_default['experiments_path'],
                               help=parameters_help['experiments_path'])

        # ------------ #
        # LOAD DATASET #
        # ------------ #
        subparser.add_argument('--dataset',
                               type=str,
                               default=parameters_default['dataset'],
                               help=parameters_help['dataset'])

        subparser.add_argument('--image_height',
                               type=int,
                               default=parameters_default['image_height'],
                               help=parameters_help['image_height'])

        subparser.add_argument('--image_width',
                               type=int,
                               default=parameters_default['image_width'],
                               help=parameters_help['image_width'])

        subparser.add_argument('--split',
                               type=str,
                               default=parameters_default['split'],
                               help=parameters_help['split'])

        subparser.add_argument('--num_classes',
                               type=int,
                               default=parameters_default['num_classes'],
                               help=parameters_help['num_classes'])

        # ------------- #
        # EXPERIMENT ID #
        # ------------- #
        subparser.add_argument('--typeID',
                               type=str,
                               default=parameters_default['typeID'],
                               help=parameters_help['typeID'])

        subparser.add_argument('--sep',
                               type=str,
                               default=parameters_default['sep'],
                               help=parameters_help['sep'])

        # ------ #
        # DEVICE #
        # ------ #
        subparser.add_argument('--GPU',
                               type=str,
                               default=parameters_default['GPU'],
                               help=parameters_help['GPU'])

        subparser.add_argument('--num_threads',
                               type=int,
                               default=parameters_default['num_threads'],
                               help=parameters_help['num_threads'])

        # --------------- #
        # REPRODUCIBILITY #
        # --------------- #
        subparser.add_argument('--seed',
                               type=int,
                               default=parameters_default['seed'],
                               help=parameters_help['seed'])

        # --------------------- #
        # DATASET NORMALIZATION #
        # --------------------- #
        subparser.add_argument('--norm',
                               type=str,
                               default=parameters_default['norm'],
                               help=parameters_help['norm'])

        # ------------------ #
        # DATASET TRANSFORMS #
        # ------------------ #
        subparser.add_argument('--resize',
                               type=int,
                               default=parameters_default['resize'],
                               help=parameters_help['resize'])

        # ----------- #
        # DATA LOADER #
        # ----------- #
        subparser.add_argument('--batch_size_train', '--bs',
                               type=int,
                               default=parameters_default['batch_size_train'],
                               help=parameters_help['batch_size_train'])

        subparser.add_argument('--batch_size_val',
                               type=int,
                               default=parameters_default['batch_size_val'],
                               help=parameters_help['batch_size_val'])

        subparser.add_argument('--batch_size_test',
                               type=int,
                               default=parameters_default['batch_size_test'],
                               help=parameters_help['batch_size_test'])

        subparser.add_argument('--num_workers',
                               type=int,
                               default=parameters_default['num_workers'],
                               help=parameters_help['num_workers'])

        # ------- #
        # NETWORK #
        # ------- #
        subparser.add_argument('--model',
                               type=str,
                               default=parameters_default['model'],
                               choices=parameters_choices['model'],
                               help=parameters_help['model'])

        subparser.add_argument('--pretrained',
                               action='store_true',
                               default=parameters_default['pretrained'],
                               help=parameters_help['pretrained'])

        # ---------------- #
        # HYPER-PARAMETERS #
        # ---------------- #
        subparser.add_argument('--epochs', '--ep',
                               type=int,
                               default=parameters_default['epochs'],
                               help=parameters_help['epochs'])

        subparser.add_argument('--epoch_to_resume', '--ep_to_resume',
                               type=int,
                               default=parameters_default['epoch_to_resume'],
                               help=parameters_help['epoch_to_resume'])

        subparser.add_argument('--optimizer',
                               type=str,
                               default=parameters_default['optimizer'],
                               choices=parameters_choices['optimizer'],
                               help=parameters_help['optimizer'])

        subparser.add_argument('--scheduler',
                               type=str,
                               default=parameters_default['scheduler'],
                               choices=parameters_choices['scheduler'],
                               help=parameters_help['scheduler'])

        subparser.add_argument('--clip_gradient',
                               action='store_true',
                               default=parameters_default['clip_gradient'],
                               help=parameters_help['clip_gradient'])

        subparser.add_argument('--learning_rate', '--lr',
                               type=float,
                               default=parameters_default['learning_rate'],
                               help=parameters_help['learning_rate'])

        subparser.add_argument('--lr_momentum',
                               type=int,
                               default=parameters_default['lr_momentum'],
                               help=parameters_help['lr_momentum'])

        subparser.add_argument('--lr_patience',
                               type=int,
                               default=parameters_default['lr_patience'],
                               help=parameters_help['lr_patience'])

        subparser.add_argument('--lr_step_size',
                               type=int,
                               default=parameters_default['lr_step_size'],
                               help=parameters_help['lr_step_size'])

        subparser.add_argument('--lr_gamma',
                               type=float,
                               default=parameters_default['lr_gamma'],
                               help=parameters_help['lr_gamma'])

        subparser.add_argument('--lr_T0',
                               type=int,
                               default=parameters_default['lr_T0'],
                               help=parameters_help['lr_T0'])

        subparser.add_argument('--lr_Tmax',
                               type=int,
                               default=parameters_default['lr_Tmax'],
                               help=parameters_help['lr_Tmax'])

        subparser.add_argument('--max_norm',
                               type=float,
                               default=parameters_default['max_norm'],
                               help=parameters_help['max_norm'])

        # ---- #
        # LOSS #
        # ---- #
        subparser.add_argument('--loss',
                               type=str,
                               default=parameters_default['loss'],
                               help=parameters_help['loss'])

        subparser.add_argument('--gamma',
                               type=float,
                               default=parameters_default['gamma'],
                               help=parameters_help['gamma'])

        # ---------- #
        # LOAD MODEL #
        # ---------- #
        subparser.add_argument('--load_best_accuracy_model',
                               action='store_true',
                               default=parameters_default['load_best_accuracy_model'],
                               help=parameters_help['load_best_accuracy_model'])

        # ------ #
        # OUTPUT #
        # ------ #
        subparser.add_argument('--num_images',
                               type=int,
                               default=parameters_default['num_images'],
                               help=parameters_help['num_images'])

    parser = parser.parse_args()

    return parser
