from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def show_metrics_train(metrics: dict):
    """
    Show metrics-train

    :param metrics: metrics dictionary
    """

    # metrics round
    ticks = metrics['ticks'][-1]
    loss = my_round_value(value=metrics['loss'][-1], digits=3)
    accuracy = my_round_value(value=metrics['accuracy'][-1], digits=3)
    MCC = my_round_value(value=metrics['MCC'][-1], digits=3)
    learning_rate = metrics['learning_rate'][-1]
    ROC_AUC = my_round_value(value=metrics['ROC_AUC'][-1], digits=3)
    PR_AUC = my_round_value(value=metrics['PR_AUC'][-1], digits=3)

    # metrics timer conversion
    metrics_time_train = timer(time_elapsed=metrics['time']['train'][-1])
    metrics_time_validation = timer(time_elapsed=metrics['time']['validation'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    print("\n--------------"
          "\nMETRICS TRAIN:"
          "\n--------------"
          "\nEpoch: {} | Loss: {}".format(ticks, loss))

    print("\n-------------------"
          "\nMETRICS VALIDATION:"
          "\n-------------------"
          "\nAccuracy: {}".format(accuracy),
          "\nMCC: {}".format(MCC),
          "\nROC AUC: {}".format(ROC_AUC),
          "\nPR AUC: {}".format(PR_AUC))

    print("\n-----"
          "\nTIME:"
          "\n-----"
          "\nTRAIN: {} h {} m {} s".format(metrics_time_train['hours'], metrics_time_train['minutes'], metrics_time_train['seconds']),
          "\nVALIDATION: {} h {} m {} s".format(metrics_time_validation['hours'], metrics_time_validation['minutes'], metrics_time_validation['seconds']),
          "\nMETRICS: {} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds']))
