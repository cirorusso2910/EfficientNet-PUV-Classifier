from typing import Tuple


def get_corner_coords(image_shape: Tuple[int, int],
                      corner: str) -> Tuple[int, int]:
    """
    Get corner coordinates according to corner

    :param image_shape: image shape [H, W]
    :param corner: corner [tl, tr, bl, br]
    :return: corner coordinates
    """

    # image shape
    height, width = image_shape

    # corner coordinates dictionary
    corner_coords_dict = {
        "tl": (10, 20),  # Top left
        "tr": (width - 100, 20),  # Top right
        "bl": (10, height - 20),  # Bottom left
        "br": (width - 100, height - 20)  # Bottom right
    }

    return corner_coords_dict[corner]
