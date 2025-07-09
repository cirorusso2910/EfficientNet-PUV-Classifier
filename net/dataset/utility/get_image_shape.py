import sys

import numpy as np

from net.utility.msg.msg_error import msg_error


def get_image_shape() -> np.ndarray:
    """
    Get image shape according to dataset

    :return: image shape (H x W)
    """

    # image shape
    image_shape = (224, 224)  # resize

    return np.array((int(image_shape[0]), int(image_shape[1])))
