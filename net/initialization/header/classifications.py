from typing import List


def classifications_header() -> List[str]:
    """
    Get classifications header

    :return: header
    """

    header = ["FILENAME",
              "PREDICTION",
              "SCORE",
              "LABEL"]

    return header
