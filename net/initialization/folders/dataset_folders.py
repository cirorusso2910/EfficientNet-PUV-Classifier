def dataset_folders_dict() -> dict:
    """
    Example of dataset folders dictionary
    :return: folders dictionary
    """

    dataset_folders = {
        'images': 'images',
        'images_subfolder': {
            'all': 'all',
            'label': 'label',
        },

        'info': 'info',
        'lists': 'lists',
        'split': 'split',
        'statistics': 'statistics'
    }

    return dataset_folders
