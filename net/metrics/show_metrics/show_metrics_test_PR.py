from net.metrics.utility.my_round_value import my_round_value


def show_metrics_test_PR(metrics: dict):
    """
    Show metrics-test precision-recall

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

    print("\n------------------------------"
          "\nMETRICS TEST PRECISION-RECALL:"
          "\n------------------------------"
          "\nPrecision (micro): {}".format(precision_micro),
          "\nRecall (micro): {}".format(recall_micro),
          "\nF1 (micro): {}".format(F1_micro),
          "\n"
          "\nPrecision (macro): {}".format(precision_macro),
          "\nRecall (macro): {}".format(recall_macro),
          "\nF1 (macro): {}".format(F1_macro),
          "\n"
          "\nPrecision (weighted): {}".format(precision_weighted),
          "\nRecall (weighted): {}".format(recall_weighted),
          "\nF1 (weighted): {}".format(F1_weighted))
