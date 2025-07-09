import os
from typing import List, Any

from pandas import read_csv
from skimage import io
from torch.utils.data import Dataset


class dataset_class(Dataset):
    """
    Dataset Class
    """

    def __init__(self,
                 images_dir: str,
                 images_label_dir: str,
                 filename_list: List[str],
                 transforms: Any):
        """
        __init__ method: run one when instantiating the object

        :param images_dir: images directory
        :param images_label_dir: images label directory
        :param filename_list: filename list
        :param transforms: transforms dataset to apply
        """

        self.images_dir = images_dir
        self.images_label_dir = images_label_dir
        self.filename_list = filename_list
        self.transforms = transforms

    def __len__(self) -> int:
        """
        __len__ method: returns the number of samples in dataset
        :return: number of samples in dataset
        """

        return len(self.filename_list)

    def __getitem__(self,
                    idx: int) -> dict:
        """
        __getitem__ method: loads and return a sample from the dataset at given index

        :param idx: sample index
        :return: sample dictionary
        """

        # ----- #
        # IMAGE #
        # ----- #
        image_filename = self.filename_list[idx] + ".png"
        image_path = os.path.join(self.images_dir, image_filename)
        image = io.imread(image_path)  # numpy.ndarray [HxWxC - 4-channels image or HxW]
        # check channels image
        if image.ndim == 3 and image.shape[2] >= 1:
            image = image[:, :, 0]  # first channel

        # ----------- #
        # IMAGE LABEL #
        # ----------- #
        images_label = read_csv(filepath_or_buffer=self.images_label_dir, usecols=["CLASS"]).values  # numpy.ndarray
        image_label = images_label[idx][0]  # single element array

        sample = {'filename': self.filename_list[idx],
                  'image': image,
                  'label': image_label
                  }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
