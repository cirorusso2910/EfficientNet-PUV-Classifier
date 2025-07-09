import os


def experiment_result_path_dict(experiment_path: str,
                                experiment_folders: dict) -> dict:
    """
    Concatenate experiment result path

    :param experiment_path: experiment path
    :param experiment_folders: experiment folders dictionary
    :return: experiment result path dictionary
    """

    # classifications
    classifications_path = os.path.join(experiment_path, experiment_folders['classifications'])

    # log
    log_path = os.path.join(experiment_path, experiment_folders['log'])

    # metrics test
    metrics_test_path = os.path.join(experiment_path, experiment_folders['metrics_test'])

    # metrics train
    metrics_train_path = os.path.join(experiment_path, experiment_folders['metrics_train'])

    # models
    models_path = os.path.join(experiment_path, experiment_folders['models'])

    # output
    output_path = os.path.join(experiment_path, experiment_folders['output'])

    # output test
    output_test_path = os.path.join(output_path, experiment_folders['output_test'])

    # plots test
    plots_test_path = os.path.join(experiment_path, experiment_folders['plots_test'])
    coords_test_path = os.path.join(plots_test_path, experiment_folders['coords_test'])

    # plots train
    plots_train_path = os.path.join(experiment_path, experiment_folders['plots_train'])

    # plots validation
    plots_validation_path = os.path.join(experiment_path, experiment_folders['plots_validation'])

    experiment_result_path = {
        'classifications': classifications_path,

        'log': log_path,

        'metrics_test': metrics_test_path,
        'metrics_train': metrics_train_path,
        'models': models_path,

        'output': output_path,
        'output_test': output_test_path,

        'plots_test': plots_test_path,
        'coords_test': coords_test_path,

        'plots_train': plots_train_path,

        'plots_validation': plots_validation_path,
    }

    return experiment_result_path
