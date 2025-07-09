import sys

from net.model.MyEfficientNet_models import MyEfficientNet_models
from net.utility.msg.msg_error import msg_error


def MyModel(model: str,
            pretrained: bool,
            num_classes: int):
    """
    My Model

    :param model: model name
    :param pretrained: pretrained flag
    :param num_classes: number of classes
    :return: my network model
    """

    # ------------ #
    # EfficientNet #
    # ------------ #
    if model.split('-')[0] == 'EfficientNet':

        # EfficientNet [B0, B1, B2, B3, B4, B5, B6, B7]
        efficientnet = str(model.split('-')[1])

        # EfficientNet Model
        net = MyEfficientNet_models(efficientnet=efficientnet,
                                    num_classes=num_classes,
                                    pretrained=pretrained)

    else:
        str_err = msg_error(file=__file__,
                            variable=model,
                            type_variable="model name",
                            choices="[EfficientNet]")
        sys.exit(str_err)

    return net
