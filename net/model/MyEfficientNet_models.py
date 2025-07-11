import sys
from typing import Tuple, Union

from torch.nn import Linear
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights, EfficientNet_B7_Weights

from net.model.EfficientNet.MyEfficientNetB0 import MyEfficientNetB0
from net.model.EfficientNet.MyEfficientNetB1 import MyEfficientNetB1
from net.model.EfficientNet.MyEfficientNetB2 import MyEfficientNetB2
from net.model.EfficientNet.MyEfficientNetB3 import MyEfficientNetB3
from net.model.EfficientNet.MyEfficientNetB4 import MyEfficientNetB4
from net.model.EfficientNet.MyEfficientNetB5 import MyEfficientNetB5
from net.model.EfficientNet.MyEfficientNetB6 import MyEfficientNetB6
from net.model.EfficientNet.MyEfficientNetB7 import MyEfficientNetB7
from net.utility.msg.msg_error import msg_error


def MyEfficientNet_models(efficientnet: str,
                          num_classes: int,
                          pretrained: bool) -> Tuple[MyEfficientNetB0, MyEfficientNetB1, MyEfficientNetB2, MyEfficientNetB3, MyEfficientNetB4, MyEfficientNetB5, MyEfficientNetB6, MyEfficientNetB7]:
    """
    Get EfficientNet models

    :param efficientnet: EfficientNet [B0, B1, B2, B3 ,B4, B5, B6, B7]
    :param num_classes: number of classes
    :param pretrained: pretrained flag
    :return: EfficientNet model,
             num features
    """

    # --------------- #
    # EfficientNet-B0 #
    # --------------- #
    if efficientnet == 'B0':
        EfficientNet_model = MyEfficientNetB0()  # MyEfficientNet-B0 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B1 #
    # --------------- #
    elif efficientnet == 'B1':
        EfficientNet_model = MyEfficientNetB1()  # MyEfficientNet-B1 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B2 #
    # --------------- #
    elif efficientnet == 'B2':
        EfficientNet_model = MyEfficientNetB2()  # MyEfficientNet-B2 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B3 #
    # --------------- #
    elif efficientnet == 'B3':
        EfficientNet_model = MyEfficientNetB3()  # MyEfficientNet-B3 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B4 #
    # --------------- #
    elif efficientnet == 'B4':
        EfficientNet_model = MyEfficientNetB4()  # MyEfficientNet-B4 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B5 #
    # --------------- #
    elif efficientnet == 'B5':
        EfficientNet_model = MyEfficientNetB5()  # MyEfficientNet-B5 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B6 #
    # --------------- #
    elif efficientnet == 'B6':
        EfficientNet_model = MyEfficientNetB6()  # MyEfficientNet-B6 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1).state_dict())

    # --------------- #
    # EfficientNet-B7 #
    # --------------- #
    elif efficientnet == 'B7':
        EfficientNet_model = MyEfficientNetB7()  # MyEfficientNet-B7 model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1).state_dict())

    else:
        str_err = msg_error(file=__file__,
                            variable=efficientnet,
                            type_variable="EfficientNet",
                            choices="[B0, B1, B2, B3, B4, B5, B6, B7]")
        sys.exit(str_err)

    # ----------------------- #
    # CUSTOMIZED ARCHITECTURE #
    # ----------------------- #
    # overload fc layer for num-class output
    EfficientNet_model.classifier[1] = Linear(in_features=EfficientNet_model.classifier[1].in_features, out_features=num_classes)

    return EfficientNet_model
