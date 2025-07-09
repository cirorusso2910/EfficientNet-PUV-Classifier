import torch


class ToTensor(object):
    """
    ToTensor: convert nd arrays to tensor
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        image = sample['image']

        image = torch.from_numpy(image.copy()).float()

        sample = {'filename': sample['filename'],
                  'image': image,
                  'label': sample['label']
                  }

        return sample
