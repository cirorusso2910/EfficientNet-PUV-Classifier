from pandas import read_csv

from net.initialization.header.class_statistics import class_statistics_header


def read_class_statistics(statistics_path: str,
                          num_classes: int) -> dict:
    """
    Read class statistics

    :param statistics_path: statistics path
    :param num_classes: num classes
    :return: statistics dictionary
    """

    # get class statistics header
    header = class_statistics_header(num_classes=num_classes)

    header_to_remove = ["DATASET", "IMAGES", "MIN", "MAX", "MEAN", "STD"]
    labels = [item for item in header if item not in header_to_remove]

    # read class statistics
    statistics = read_csv(filepath_or_buffer=statistics_path, usecols=header)

    statistics_dict = {
        'images': {
            'train': statistics['IMAGES'][0],
            'validation': statistics['IMAGES'][1],
            'test': statistics['IMAGES'][2],
        },
        'label': {}
    }

    # for each label
    for label in labels:
        # if there is label in statistics
        if label in statistics:
            # add to dict
            statistics_dict['label'][label] = {
                'train': statistics[label][0],
                'validation': statistics[label][1],
                'test': statistics[label][2],
            }

    return statistics_dict
