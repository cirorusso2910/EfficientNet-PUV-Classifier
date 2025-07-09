from torch.utils.data import Dataset


def get_num_label(dataset: Dataset,
                  label: dict) -> dict:
    """
    Get num label

    :param dataset: dataset
    :param label: label dictionary
    :return: num label per class (positive / negative)
    """

    # dataset size
    dataset_size = dataset.__len__()

    # create a mapping dictionary between 'class' and 'label'
    class_to_label = dict(zip(label['class'], label['label']))
    # reverse mapping: from label to class
    label_to_class = {v: k for k, v in class_to_label.items()}

    # initialize a dictionary to store the count of labels for each class
    labels_per_class = {class_name: 0 for class_name in class_to_label.keys()}

    # for each sample
    for i in range(dataset_size):

        # get label
        label = dataset[i]['label']

        # get the corresponding label to class
        class_name = label_to_class[label]

        # increment the count for this class
        labels_per_class[class_name] += 1

    return labels_per_class
