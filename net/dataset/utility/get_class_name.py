from net.utility.read_label import read_label


def get_class_name(label_path: str,
                   label_to_find: int) -> str:
    """
    Get class name according to class label

    :param label_path: label path
    :param label_to_find: label to find
    :return: class name
    """

    # read label
    label = read_label(label_path=label_path)

    # create an inverse mapping dictionary
    label_to_class = {label: cls for cls, label in zip(label['class_name'], label['label'])}

    # find the corresponding class
    class_name = label_to_class.get(label_to_find)

    return class_name
