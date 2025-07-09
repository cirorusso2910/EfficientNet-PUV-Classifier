import csv
import os

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value


def metrics_test_PR_csv(metrics_PR_path: str,
                        metrics: dict):
    """
    Save metrics-test-PR.csv

    :param metrics_PR_path: metrics PR path
    :param metrics: metrics dictionary
    """

    # metrics round
    precision_micro = my_round_value(value=metrics['precision']['micro'][-1], digits=3)
    recall_micro = my_round_value(value=metrics['recall']['micro'][-1], digits=3)
    F1_micro = my_round_value(value=metrics['F1']['micro'][-1], digits=3)

    precision_macro = my_round_value(value=metrics['precision']['macro'][-1], digits=3)
    recall_macro = my_round_value(value=metrics['recall']['macro'][-1], digits=3)
    F1_macro = my_round_value(value=metrics['F1']['macro'][-1], digits=3)

    precision_weighted = my_round_value(value=metrics['precision']['weighted'][-1], digits=3)
    recall_weighted = my_round_value(value=metrics['recall']['weighted'][-1], digits=3)
    F1_weighted = my_round_value(value=metrics['F1']['weighted'][-1], digits=3)

    # check if file exists
    file_exists = os.path.isfile(metrics_PR_path)
    # if metrics already exists: delete
    if file_exists:
        os.remove(metrics_PR_path)

    # save metrics-test-PR.csv
    with open(metrics_PR_path, 'a') as file:
        # writer
        writer = csv.writer(file)

        if not file_exists:
            # write header
            header = metrics_header(metrics_type='test_PR')
            writer.writerow(header)

        # write row
        writer.writerow(["MICRO",
                         precision_micro,
                         recall_micro,
                         F1_micro])

        # write row
        writer.writerow(["MACRO",
                         precision_macro,
                         recall_macro,
                         F1_macro])

        # write row
        writer.writerow(["WEIGHTED",
                         precision_weighted,
                         recall_weighted,
                         F1_weighted])
