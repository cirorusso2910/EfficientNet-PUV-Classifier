from typing import Tuple

import numpy as np
import torch

from torchvision.transforms import transforms, InterpolationMode

from net.dataset.draw.add_3_channels_image import add_3_channels_image


class Resize(object):
    """
    Resize image and annotation according to size (H x W) and tool (PyTorch, OpenCV)
    """

    def __init__(self,
                 size: Tuple[int, int],
                 num_channels: int):
        """
        __init__ method: run one when instantiating the object

        :param size: image size
        :param num_channels: image num channels
        """

        self.size = size

        self.num_channels = num_channels

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read sample
        image = sample['image']

        # get image shape
        if self.num_channels == 1:
            # image: (H x W) -> (H x W x C)
            image = add_3_channels_image(image=image)  # H x W x C
            image = np.transpose(image, (2, 0, 1))  # HxWxC -> CxHxW

        # transforms torchvision resize
        transforms_resize = transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST, max_size=None, antialias=None)

        # resize image
        image = torch.from_numpy(image)  # numpy to tensor
        image_resized = transforms_resize(image)
        image_resized = image_resized.cpu().detach().numpy()  # tensor to numpy

        sample = {'filename': sample['filename'],
                  'image': image_resized,
                  'label': sample['label']
                  }

        return sample
