import sys
from typing import List

from net.utility.msg.msg_error import msg_error


def class_statistics_header(num_classes: int) -> List:
    """
    Get class statistics header

    :param num_classes: num classes
    :return: header
    """

    # 2-classes
    if num_classes == 2:
        header = ["DATASET",
                  "IMAGES",
                  "POSITIVE",  # -> CLASS 0
                  "NEGATIVE",  # -> CLASS 1
                  "MIN",
                  "MAX",
                  "MEAN",
                  "STD"]

    else:
        str_err = msg_error(file=__file__,
                            variable=num_classes,
                            type_variable="num classes (SUD4VUP)",
                            choices="[2]")
        sys.exit(str_err)

    return header
