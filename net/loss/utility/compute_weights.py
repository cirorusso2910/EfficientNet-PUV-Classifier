import torch


def compute_weights(num_label: dict):
    """
    Compute weights per class

    :param num_label: num label dictionary
    :return: weights per class
    """

    # sum total dataset sample
    total_sample = sum(num_label.values())

    # init
    weights = []

    # compute weight for each class
    for class_label, count in num_label.items():
        weight_value = count / total_sample
        weights.append(weight_value)

    return torch.Tensor(weights)


def compute_weights_normalized(num_label: dict) -> torch.Tensor:
    """
    Compute weights normalized per class

    :param num_label: num label dictionary
    :return: weights per class
    """

    # init
    weights = []

    # compute inverse weight for each class
    for class_label, count in num_label.items():
        weight_value = 1 / count  # Inversely proportional to the class count
        weights.append(weight_value)

    # Normalize the weights to sum to 1 (optional but helps with stability)
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Return as a Tensor
    return torch.Tensor(normalized_weights)
