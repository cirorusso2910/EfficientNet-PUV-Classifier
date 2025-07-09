def get_num_label_dict(statistics: dict) -> dict:
    """
    Process class statistics to obtain num label dictionary

    :param statistics: statistics dictionary
    :return: num label dictionary
    """

    # get all class labels
    class_labels = list(statistics['label'].keys())

    # init dict
    num_label = {
        'train': {},
        'validation': {},
        'test': {},
        'all': {}
    }

    # fore ach class
    for class_label in class_labels:
        num_label['train'][class_label] = statistics['label'][class_label]['train']
        num_label['validation'][class_label] = statistics['label'][class_label]['validation']
        num_label['test'][class_label] = statistics['label'][class_label]['test']

        num_label['all'][class_label] = (
                statistics['label'][class_label]['train'] +
                statistics['label'][class_label]['validation'] +
                statistics['label'][class_label]['test']
        )

    return num_label
