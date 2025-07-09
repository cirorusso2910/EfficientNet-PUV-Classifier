import csv

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value
from net.utility.read_label import read_label


def metrics_test_class_csv(metrics_class: dict,
                           label_path: str,
                           metrics_test_class_path: str):
    """
    Save metrics-test-class.csv

    :param metrics_class: metrics class dictionary
    :param label_path: label path
    :param metrics_test_class_path: metrics-test-class path
    """

    label = read_label(label_path=label_path)
    label_class = label['class']

    # save metrics-test-class.csv
    with open(metrics_test_class_path, 'w') as file:
        # writer
        writer = csv.writer(file)

        header = metrics_header(metrics_type='test_class')
        writer.writerow(header)

        # for each class
        for i in range(len(label_class)):

            writer.writerow([label_class[i],
                             my_round_value(value=metrics_class['precision'][i], digits=3),
                             my_round_value(value=metrics_class['recall'][i], digits=3),
                             my_round_value(value=metrics_class['F1'][i], digits=3)])
