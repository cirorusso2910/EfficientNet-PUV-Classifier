def msg_load_best_model_complete(metrics_type: str,
                                 load_model: dict):
    """
    Message load best model complete

    :param metrics_type: metrics type
    :param load_model: load model dictionary
    """

    print("LOADED BEST MODEL ({}):".format(metrics_type.upper()),
          "\ntrained for {} epochs with {:.3f}".format(load_model['epoch'],
                                                       load_model[metrics_type]))


def msg_load_resume_model_complete(load_model: dict):
    """
    Message load resume model complete

    :param load_model: load model dictionary
    """

    print("LOADED RESUME-MODEL:"
          "\ntrained for {} epochs with".format(load_model['epoch']),
          "\n- Accuracy: {:.3f}".format(load_model['accuracy']),
          "\n- MCC: {:.3f}".format(load_model['MCC']),
          "\n- ROC AUC: {:.3f}".format(load_model['ROC_AUC']),
          "\n- PR AUC: {:.3f}".format(load_model['PR_AUC']))
