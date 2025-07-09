import os
import time

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from net.classifications.classifications_test import classifications_test


def test(net: torch.nn.Module,
         dataloader: DataLoader,
         classifications_path: str,
         device: torch.device
         ):
    """
    Test function

    :param net: net
    :param dataloader: dataloader
    :param classifications_path: classifications path
    :param device: device
    """
    # switch to test mode
    net.eval()

    # if classifications already exists: delete
    if os.path.isfile(classifications_path):
        os.remove(classifications_path)

    # do not accumulate gradients (faster)
    with torch.no_grad():
        # for each batch in dataloader
        for num_batch, batch in enumerate(dataloader):
            # init batch time
            time_batch_start = time.time()

            # get data from dataloader
            filename, image, label = batch['filename'], batch['image'].float().to(device), batch['label'].to(device)

            # forward pass
            classifications = net(image)

            # compute predicted probabilities apply
            classifications_probabilities = F.softmax(classifications, dim=1)
            # convert outputs to predicted class (argmax for classification)
            predicted_labels = classifications_probabilities.max(dim=1)[1]  # labels with max score per row
            predicted_score = classifications_probabilities.max(dim=1)[0]  # max score per row

            # save classifications.csv
            classifications_test(filenames=filename,
                                 labels=label,
                                 predicted_labels=predicted_labels,
                                 predicted_score=predicted_score,
                                 classifications_path=classifications_path)

            # batch time
            time_batch = time.time() - time_batch_start

            # show
            print("Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
                  "Time: {:.0f} s ".format(int(time_batch) % 60))
