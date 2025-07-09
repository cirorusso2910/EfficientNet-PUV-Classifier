from net.metrics.utility.my_round_value import my_round_value
from net.utility.read_label import read_label


def show_metrics_test_class(metrics_class: dict,
                            label_path: str):
    """
    Show metrics-test-class

    :param metrics_class: metrics class dictionary
    :param label_path: label path
    """

    label = read_label(label_path=label_path)
    label_class = label['class']
    num_classes = len(label_class)

    print("\n-------------------"
          "\nMETRICS TEST CLASS:"
          "\n-------------------")

    for i in range(num_classes):
        print("CLASS: {}".format(label_class[i]),
              "\nPrecision: {} | Recall: {} | F1: {}".format(my_round_value(value=metrics_class['precision'][i], digits=3),
                                                             my_round_value(value=metrics_class['recall'][i], digits=3),
                                                             my_round_value(value=metrics_class['F1'][i], digits=3)))
