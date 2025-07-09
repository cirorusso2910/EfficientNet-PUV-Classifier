import cv2
from torch.utils.data import Dataset

from net.dataset.utility.viewable_image import viewable_image


def read_dataset_sample(dataset: Dataset,
                        idx: int) -> dict:
    """
    Read dataset sample at specific index (idx) position

    :param dataset: dataset
    :param idx: index
    :return: sample dictionary
    """

    # image filename
    image_filename = dataset[idx]['filename']

    # image
    image = dataset[idx]['image']
    image = image.permute((1, 2, 0))  # permute
    image = image.cpu().detach().numpy()
    image = image.copy()
    image = viewable_image(image=image)

    # convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image label
    image_label = dataset[idx]['label']

    sample = {
        'filename': image_filename,
        'image': image,
        'label': image_label
    }

    return sample
