def experiment_folders_dict() -> dict:
    """
    Experiment folders dictionary

    :return: folders dictionary
    """

    experiment_result_folders = {
        'classifications': 'classifications',
        'log': 'log',
        'metrics_test': 'metrics-test',
        'metrics_train': 'metrics-train',
        'models': 'models',

        'output': 'output',
        'output_test': 'output-test',

        'plots_test': 'plots-test',
        'coords_test': 'coords',

        'plots_train': 'plots-train',

        'plots_validation': 'plots-validation',
    }

    return experiment_result_folders
