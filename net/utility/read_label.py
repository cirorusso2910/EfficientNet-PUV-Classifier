from pandas import read_csv


def read_label(label_path: str) -> dict:
    """
    Read data label

    :param label_path: path label
    :return: label dictionary
    """

    # read csv
    label_csv = read_csv(filepath_or_buffer=label_path, usecols=["CLASS", "CLASS NAME", "LABEL"]).values

    class_name = label_csv[:, 0]
    class_name_output = label_csv[:, 1]
    label = label_csv[:, 2]

    label_dict = {
        'class': class_name.tolist(),
        'class_name': class_name_output.tolist(),
        'label': label.tolist(),
    }

    return label_dict
