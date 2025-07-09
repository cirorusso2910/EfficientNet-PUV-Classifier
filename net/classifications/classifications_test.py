import os
from typing import List

import pandas as pd
import torch

from net.initialization.header.classifications import classifications_header


def classifications_test(filenames: List[str],
                         labels: torch.Tensor,
                         predicted_labels: torch.Tensor,
                         predicted_score: torch.Tensor,
                         classifications_path: str):
    """
    Compute classifications in test and save in classifications.csv

    :param filenames: filenames
    :param labels: labels
    :param predicted_labels: predicted labels
    :param predicted_score: predicted score
    :param classifications_path: classifications path
    """

    # init
    labels = labels.tolist()
    predicted_labels = predicted_labels.tolist()
    predicted_score = predicted_score.tolist()

    # --------- #
    # DATA ROWS #
    # --------- #
    data_rows = [[filenames[i], predicted_labels[i], predicted_score[i], labels[i]] for i in range(len(filenames))]

    # define DataFrame
    df = pd.DataFrame(data_rows)

    # -------------------- #
    # SAVE CLASSIFICATIONS #
    # -------------------- #
    if not os.path.exists(classifications_path):
        df.to_csv(path_or_buf=classifications_path, mode='a', index=False, header=classifications_header(), float_format='%g')  # write header
    else:
        df.to_csv(path_or_buf=classifications_path, mode='a', index=False, header=False, float_format='%g')  # write without header
