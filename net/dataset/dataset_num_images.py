import os
import time

from torch.utils.data import Dataset

from net.dataset.statistics.read_class_statistics import read_class_statistics


def dataset_num_images(statistics_path: str,
                       dataset_name: str,
                       num_classes: int,
                       dataset_train: Dataset,
                       dataset_val: Dataset,
                       dataset_test: Dataset) -> dict:
    """
    Get dataset num images

    :param statistics_path: statistics path
    :param dataset_name: dataset name
    :param num_classes: num classes
    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: dataset num images dictionary
    """

    # if statistics exists
    if os.path.exists(statistics_path):

        # read statistics
        statistics = read_class_statistics(statistics_path=statistics_path,
                                           num_classes=num_classes)

        num_images = {
            'train': statistics['images']['train'],
            'validation': statistics['images']['validation'],
            'test': statistics['images']['test']
        }

    # if no statistics exists
    else:

        time_images_start = time.time()

        num_images_train = dataset_train.__len__()
        num_images_val = dataset_val.__len__()
        num_images_test = dataset_test.__len__()

        time_images = time.time() - time_images_start

        print("Num images computed in: {} m {} s".format(int(time_images) // 60, int(time_images) % 60))

        num_images = {
            'train': num_images_train,
            'validation': num_images_val,
            'test': num_images_test
        }

    return num_images
