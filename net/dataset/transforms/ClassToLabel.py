class ClassToLabel(object):
    """
    ClassToLabel: convert class string to label int
    """

    def __init__(self,
                 label: dict):
        """
        __init__ method: run one when instantiating the object

        :param label: class-label dict
        """

        self.label = label

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read label
        label = sample['label']

        # create a mapping dictionary between 'class' and 'label'
        class_to_label = dict(zip(self.label['class'], self.label['label']))

        if label in class_to_label:
            image_label = class_to_label[label]

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'label': image_label,
                  }

        return sample
