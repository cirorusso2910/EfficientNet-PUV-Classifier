import os

import cv2

from pandas import read_csv
from torch.utils.data import Dataset

from net.colors.colors import *
from net.dataset.utility.get_class_name import get_class_name
from net.dataset.utility.read_dataset_sample import read_dataset_sample
from net.output.utility.get_corner_coords import get_corner_coords


def output(dataset: Dataset,
           num_images: int,
           classifications_path: str,
           label_path: str,
           output_path: str,
           suffix: str):
    """
    Save detections output results

    :param type_draw: type draw
    :param eval: evaluation type
    :param box_draw_radius: box draw radius
    :param dataset: dataset
    :param num_images: num images
    :param detections_path: detections path
    :param classifications_path: classifications path
    :param label_path: label path
    :param output_path: output path
    :param suffix: suffix
    """

    # read classifications test
    classifications = read_csv(filepath_or_buffer=classifications_path, usecols=["PREDICTION"]).values

    # for each sample in dataset
    for i in range(dataset.__len__()):

        # read dataset sample
        sample = read_dataset_sample(dataset=dataset,
                                     idx=i)

        # image filename
        image_filename = sample['filename']

        # image label
        image_label = sample['label']

        # image label predicted
        image_label_predicted = classifications[i].item()

        # image
        image = sample['image']
        height, width, _ = image.shape  # image shape
        image_shape = (height, width)

        # ---------------- #
        # DRAW IMAGE LABEL #
        # ---------------- #
        # get corner coords
        corner_coords = get_corner_coords(image_shape=image_shape, corner='bl')
        if image_label == image_label_predicted:
            label_name_predicted = get_class_name(label_path=label_path, label_to_find=image_label_predicted)
            cv2.putText(sample['image'], text=label_name_predicted, org=corner_coords, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN2, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        else:
            label_name = get_class_name(label_path=label_path, label_to_find=image_label)
            label_name_predicted = get_class_name(label_path=label_path, label_to_find=image_label_predicted)
            cv2.putText(sample['image'], text="{} ({})".format(label_name_predicted, label_name), org=corner_coords, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=RED1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

        # -------- #
        # FILENAME #
        # -------- #
        image_output_filename = image_filename + suffix

        # image output path
        image_output_path = os.path.join(output_path, image_output_filename + ".png")

        # ---------- #
        # SAVE IMAGE #
        # ---------- #
        cv2.imwrite(image_output_path, image)
        print("Image {}/{}: {} saved".format(i + 1, num_images, image_filename))

        if num_images == i + 1:
            return
