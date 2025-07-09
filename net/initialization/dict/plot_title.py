def plot_title_dict():
    """
    Define plot title experiment results

    :return: plot title dictionary
    """

    plot_title = {
        'plots_train': {
            'loss': 'LOSS',
            'learning_rate': 'LEARNING RATE'
        },

        'plots_validation': {
            'accuracy': 'ACCURACY',
            'MCC': 'MCC',
            'ROC_AUC': 'ROC AUC',
            'PR_AUC': 'PR AUC'
        },

        'plots_test': {
            'ROC': 'ROC (TEST)',
            'PR': 'PR (TEST)',
            'score_distribution': "SCORE DISTRIBUTION (TEST)"
        }
    }

    return plot_title
