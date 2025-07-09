import sys

from typing import List


def coords_header(coords_type: str) -> List:
    """
    Get coords header according to type

    :param coords_type: coords type
    :return: header
    """

    if coords_type == 'ROC':
        header = ["FPR",
                  "TPR"]

    elif coords_type == 'PR':
        header = ["PRECISION",
                  "RECALL"]

    else:
        str_err = "\nERROR in {}" \
                  "\n{} wrong header coords type" \
                  "\n[ROC, PR]".format(__file__,
                                       coords_type)
        sys.exit(str_err)

    return header
