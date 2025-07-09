import os
import time

from torch.utils.data import Dataset, ConcatDataset

from net.dataset.statistics.num_label import get_num_label
from net.dataset.statistics.read_class_statistics import read_class_statistics
from net.dataset.utility.get_num_label_dict import get_num_label_dict
from net.utility.read_label import read_label


def dataset_num_label(statistics_path: str,
                      label_path: str,
                      num_classes: int,
                      dataset_train: Dataset,
                      dataset_val: Dataset,
                      dataset_test: Dataset) -> dict:
    """
    Get dataset num label

    :param statistics_path: statistics path
    :param label_path: label path
    :param num_classes: num classes
    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: num label per dataset
    """

    # if statistics exists
    if os.path.exists(statistics_path):

        # read statistics
        statistics = read_class_statistics(statistics_path=statistics_path,
                                           num_classes=num_classes)

        # get num label dictionary
        num_label = get_num_label_dict(statistics=statistics)

    else:

        # read label
        label = read_label(label_path=label_path)

        time_label_start = time.time()

        # num label in dataset train
        num_label_train = get_num_label(dataset=dataset_train,
                                        label=label)

        # num label in dataset validation
        num_label_validation = get_num_label(dataset=dataset_val,
                                             label=label)

        # num label in dataset test
        num_label_test = get_num_label(dataset=dataset_test,
                                       label=label)

        # num label in dataset
        dataset_concat = ConcatDataset([dataset_train, dataset_val, dataset_test])
        num_label_all = get_num_label(dataset=dataset_concat,
                                      label=label)

        time_label = time.time() - time_label_start

        print("Num label computed in: {} m {} s".format(int(time_label) // 60, int(time_label) % 60))

        num_label = {
            'train': num_label_train,
            'validation': num_label_validation,
            'test': num_label_test,
            'all': num_label_all
        }

    return num_label
