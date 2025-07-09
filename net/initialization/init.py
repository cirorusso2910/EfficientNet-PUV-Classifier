import argparse
import os

from net.initialization.folders.dataset_folders import dataset_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.path.experiment_result_path import experiment_result_path_dict
from net.initialization.utility.create_folder import create_folder
from net.initialization.utility.create_folder_and_subfolder import create_folder_and_subfolder


def initialization(network_name: str,
                   experiment_ID: str,
                   parser: argparse.Namespace) -> dict:
    """
    Initialization of experiment results folder based on execution mode

    :param network_name: network name
    :param experiment_ID: experiment ID
    :param parser: parser of parameters-parsing
    :return: path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # dataset folders
    dataset_folders = dataset_folders_dict()

    # experiment folders
    experiment_folders = experiment_folders_dict()

    # ------- #
    # DATASET #
    # ------- #
    # images
    images_all_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['images'], dataset_folders['images_subfolder']['all'])

    # images label
    images_label_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['images'], dataset_folders['images_subfolder']['label'], 'images-label-{}.csv'.format(parser.num_classes))
    images_class_label_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['images'], dataset_folders['images_subfolder']['label'], 'images-class-label-{}.csv'.format(parser.num_classes))

    # data split
    data_split_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['split'], 'split-' + parser.split + '.csv')

    # lists
    lists_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['lists'])
    list_all_path = os.path.join(str(lists_path), 'all.txt')

    # statistics
    class_statistics_filename = "split-{}-{}-class-statistics.csv".format(parser.split, parser.num_classes)
    class_statistics_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['statistics'], class_statistics_filename)

    # info
    info_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['info'])

    # path dataset dict
    path_dataset_dict = {
        'images': {
            'all': images_all_path,
            'label': images_label_path,
            'class_label': images_class_label_path
        },

        'lists': {
            'all': list_all_path,
        },

        'info': info_path,

        'split': data_split_path,

        'statistics': class_statistics_path,
    }

    # ---- #
    # PATH #
    # ---- #
    # experiment
    experiment_name = network_name + "|" + experiment_ID
    experiment_path = os.path.join(parser.experiments_path, experiment_name)

    # experiment path
    experiment_results_path = experiment_result_path_dict(experiment_path=experiment_path,
                                                          experiment_folders=experiment_folders)

    # explainability path
    explainability_folder = "explainability"
    explainability_path = os.path.join(experiment_path, explainability_folder)

    # -------------------- #
    # CREATE RESULT FOLDER #
    # -------------------- #
    # create experiment folder
    if parser.mode in ['train', 'train_test']:
        create_folder_and_subfolder(main_path=experiment_path,
                                    subfolder_path_dict=experiment_results_path)
        print("Experiment result folder: COMPLETE")

    elif parser.mode in ['test']:
        print("Experiment result folder: ALREADY COMPLETE")

    elif parser.mode in ['explainability']:
        create_folder(path=explainability_path)
        print("Explainability result folder: COMPLETE")

    else:
        print("Experiment result folder: ALREADY COMPLETE")

    # ----------- #
    # RESULT PATH #
    # ----------- #
    # classifications
    classifications_validation_filename = "classifications-validation|" + experiment_ID + ".csv"
    classifications_validation_path = os.path.join(experiment_results_path['classifications'], classifications_validation_filename)

    classifications_test_filename = "classifications-test|" + experiment_ID + ".csv"
    classifications_test_path = os.path.join(experiment_results_path['classifications'], classifications_test_filename)

    # metrics-train
    metrics_train_filename = "metrics-train|" + experiment_ID + ".csv"
    metrics_train_path = os.path.join(experiment_results_path['metrics_train'], metrics_train_filename)

    # metrics-test
    metrics_test_filename = "metrics-test|" + experiment_ID + ".csv"
    metrics_test_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_filename)

    confusion_matrix_test_filename = "confusion-matrix-test|" + experiment_ID + ".csv"
    confusion_matrix_test_path = os.path.join(experiment_results_path['metrics_test'], confusion_matrix_test_filename)

    metrics_test_class_filename = "metrics-test-class|" + experiment_ID + ".csv"
    metrics_test_class_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_class_filename)

    metrics_test_PR_filename = "metrics-test-PR|" + experiment_ID + ".csv"
    metrics_test_PR_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_PR_filename)

    # models best
    model_best_accuracy_filename = network_name + "-best-model-accuracy|" + experiment_ID + ".tar"
    model_best_accuracy_path = os.path.join(experiment_results_path['models'], model_best_accuracy_filename)

    # plots-train
    loss_filename = "Loss|" + experiment_ID + ".png"
    loss_path = os.path.join(experiment_results_path['plots_train'], loss_filename)

    # plots-validation
    accuracy_filename = "Accuracy|" + experiment_ID + ".png"
    accuracy_path = os.path.join(experiment_results_path['plots_validation'], accuracy_filename)

    MCC_filename = "MCC|" + experiment_ID + ".png"
    MCC_path = os.path.join(experiment_results_path['plots_validation'], MCC_filename)

    ROC_AUC_filename = "ROC-AUC|" + experiment_ID + ".png"
    ROC_AUC_path = os.path.join(experiment_results_path['plots_validation'], ROC_AUC_filename)

    PR_AUC_filename = "PR-AUC|" + experiment_ID + ".png"
    PR_AUC_path = os.path.join(experiment_results_path['plots_validation'], PR_AUC_filename)

    # plots-test
    ROC_test_filename = "ROC|" + experiment_ID + ".png"
    ROC_test_path = os.path.join(experiment_results_path['plots_test'], ROC_test_filename)

    PR_test_filename = "PR|" + experiment_ID + ".png"
    PR_test_path = os.path.join(experiment_results_path['plots_test'], PR_test_filename)

    score_distribution_filename = "Score-Distribution|" + experiment_ID + ".png"
    score_distribution_path = os.path.join(experiment_results_path['plots_test'], score_distribution_filename)

    # coords test
    ROC_test_coords_filename = "ROC-coords|" + experiment_ID + ".csv"
    ROC_test_coords_path = os.path.join(experiment_results_path['coords_test'], ROC_test_coords_filename)

    PR_test_coords_filename = "PR-coords|" + experiment_ID + ".csv"
    PR_test_coords_path = os.path.join(experiment_results_path['coords_test'], PR_test_coords_filename)

    path = {
        'dataset': path_dataset_dict,

        'classifications': {
            'validation': classifications_validation_path,
            'test': classifications_test_path,

        },

        'explainability': explainability_path,

        'metrics': {
            'train': metrics_train_path,
            'test': metrics_test_path,

            'confusion_matrix_test': confusion_matrix_test_path,
            'test_class': metrics_test_class_path,

            'test_PR': metrics_test_PR_path,
        },

        'model': {
            'best': {
                'accuracy': model_best_accuracy_path,
            }
        },

        'output': {
            'test': experiment_results_path['output_test'],
        },

        'plots_train': {
            'loss': loss_path,
        },

        'plots_validation': {
            'accuracy': accuracy_path,
            'MCC': MCC_path,
            'ROC_AUC': ROC_AUC_path,
            'PR_AUC': PR_AUC_path
        },

        'plots_test': {
            'ROC': ROC_test_path,
            'PR': PR_test_path,

            'coords': {
                'ROC': ROC_test_coords_path,
                'PR': PR_test_coords_path,
            },

            'score_distribution': score_distribution_path,
        }
    }

    return path
