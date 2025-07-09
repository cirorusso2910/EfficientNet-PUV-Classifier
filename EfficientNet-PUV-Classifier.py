import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time

import numpy as np
import torch
from pandas import read_csv
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from net.dataset.classes.dataset_class import dataset_class
from net.dataset.dataset_num_images import dataset_num_images
from net.dataset.dataset_num_label import dataset_num_label
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_transforms import dataset_transforms
from net.device.get_GPU_name import get_GPU_name
from net.evaluation.F1_score import F1_score, F1_score_per_class
from net.evaluation.MCC import MCC
from net.evaluation.PR import PR
from net.evaluation.PR_AUC import PR_AUC
from net.evaluation.ROC import ROC
from net.evaluation.ROC_AUC import ROC_AUC
from net.evaluation.accuracy import accuracy
from net.evaluation.confusion_matrix import my_confusion_matrix
from net.evaluation.current_learning_rate import current_learning_rate
from net.evaluation.precision import precision, precision_per_class
from net.evaluation.recall import recall, recall_per_class
from net.initialization.ID.experimentID import experimentID
from net.initialization.dict.metrics import metrics_dict
from net.initialization.dict.plot_title import plot_title_dict
from net.initialization.init import initialization
from net.loss.get_loss import get_loss
from net.metrics.confusion_matrix_test import confusion_matrix_test_csv
from net.metrics.metrics_test import metrics_test_csv
from net.metrics.metrics_test_PR import metrics_test_PR_csv
from net.metrics.metrics_test_class import metrics_test_class_csv
from net.metrics.metrics_train import metrics_train_csv
from net.metrics.show_metrics.show_metrics_test import show_metrics_test
from net.metrics.show_metrics.show_metrics_test_PR import show_metrics_test_PR
from net.metrics.show_metrics.show_metrics_test_class import show_metrics_test_class
from net.metrics.show_metrics.show_metrics_train import show_metrics_train
from net.metrics.utility.my_notation import scientific_notation
from net.model.MyModel import MyModel
from net.model.utility.load_model import load_best_model
from net.model.utility.save_model import save_best_model
from net.optimizer.get_optimizer import get_optimizer
from net.output.output import output
from net.parameters.parameters_summary import parameters_summary
from net.plot.MCC_plot import MCC_plot
from net.plot.PR_AUC_plot import PR_AUC_plot
from net.plot.PR_plot import PR_plot
from net.plot.ROC_AUC_plot import ROC_AUC_plot
from net.plot.ROC_plot import ROC_plot
from net.plot.accuracy_plot import accuracy_plot
from net.plot.loss_plot import loss_plot
from net.plot.score_distribution_plot import score_distribution_plot
from net.plot.utility.figure_size import figure_size
from net.reproducibility.reproducibility import reproducibility
from net.parameters.parameters import parameters_parsing
from net.scheduler.get_scheduler import get_scheduler
from net.test import test
from net.train import train
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_load_dataset_complete import msg_load_dataset_complete
from net.utility.read_split import read_split
from net.validation import validation


def main():
    """
        | ==================================================== |
        |                       EFFICIENTNET                   |
        |              POSTERIOR URETHRAL VALVES (PUV)         |
        |                       CLASSIFIER                     |
        | ==================================================== |

        EfficientNet for Posterior Urethral Valves (PUV) classifier

    """

    print("| ==================================================== |\n"
          "|                       EFFICIENTNET                   |\n"
          "|              POSTERIOR URETHRAL VALVES (PUV)         |\n"
          "|                       CLASSIFIER                     |\n"
          "| ==================================================== |\n")

    # ================== #
    # PARAMETERS-PARSING #
    # ================== #
    # command line parameter parsing
    parser = parameters_parsing()

    # execution mode start
    execution_mode(mode=parser.mode,
                   option='start')

    # ============== #
    # INITIALIZATION #
    # ============== #
    print("\n---------------"
          "\nINITIALIZATION:"
          "\n---------------")

    # experiment ID
    experiment_ID = experimentID(typeID=parser.typeID,
                                 sep=parser.sep,
                                 parser=parser)

    # path initialization
    path = initialization(network_name="EfficientNet-PUV-Classifier",
                          experiment_ID=experiment_ID,
                          parser=parser)

    # read data split
    data_split = read_split(path_split=path['dataset']['split'])

    # plot title
    plot_title = plot_title_dict()

    # ====== #
    # DEVICE #
    # ====== #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:ID -> run on GPU #ID
    print("GPU device name: {}".format(get_GPU_name()))

    # =============== #
    # REPRODUCIBILITY #
    # =============== #
    reproducibility(seed=parser.seed)

    # ============ #
    # LOAD DATASET #
    # ============ #
    print("\n-------------"
          "\nLOAD DATASET:"
          "\n-------------")
    dataset = dataset_class(images_dir=path['dataset']['images']['all'],
                            images_label_dir=path['dataset']['images']['label'],
                            filename_list=data_split['filename'],
                            transforms=None)
    msg_load_dataset_complete(dataset_name=parser.dataset)

    # ============= #
    # DATASET SPLIT #
    # ============= #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=data_split,
                                                             dataset=dataset)

    # ================== #
    # DATASET NUM IMAGES #
    # ================== #
    num_images = dataset_num_images(statistics_path=path['dataset']['statistics'],
                                    dataset_name=parser.dataset,
                                    num_classes=parser.num_classes,
                                    dataset_train=dataset_train,
                                    dataset_val=dataset_val,
                                    dataset_test=dataset_test)

    # ================== #
    # DATASET TRANSFORMS #
    # ================== #
    train_transforms, val_transforms, test_transforms = dataset_transforms(normalization=parser.norm,
                                                                           parser=parser,
                                                                           statistics_path=path['dataset']['statistics'],
                                                                           label_path=path['dataset']['images']['class_label'])

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # ================= #
    # DATASET NUM LABEL #
    # ================= #
    num_label = dataset_num_label(statistics_path=path['dataset']['statistics'],
                                  label_path=path['dataset']['images']['class_label'],
                                  num_classes=parser.num_classes,
                                  dataset_train=dataset_train,
                                  dataset_val=dataset_val,
                                  dataset_test=dataset_test)

    # ============ #
    # DATA LOADERS #
    # ============ #
    # dataloader-train
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=parser.batch_size_train,
                                  shuffle=True,
                                  num_workers=parser.num_workers,
                                  pin_memory=True)

    # dataloader-val
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=parser.batch_size_val,
                                shuffle=False,
                                num_workers=parser.num_workers,
                                pin_memory=True)

    # dataloader-test
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=parser.batch_size_test,
                                 shuffle=False,
                                 num_workers=parser.num_workers,
                                 pin_memory=True)

    # ============= #
    # NETWORK MODEL #
    # ============= #
    net = MyModel(model=parser.model,
                  pretrained=parser.pretrained,
                  num_classes=parser.num_classes)

    # data parallel
    net = DataParallel(module=net)

    # move net to device
    net.to(device)

    # ======= #
    # SUMMARY #
    # ======= #
    parameters_summary(parser=parser,
                       num_images=num_images,
                       num_label=num_label)

    # =========== #
    # MODE: TRAIN #
    # =========== #
    if parser.mode in ['train', 'resume', 'train_test']:

        # ========= #
        # OPTIMIZER #
        # ========= #
        optimizer = get_optimizer(net_parameters=net.parameters(),
                                  parser=parser)

        # ========= #
        # SCHEDULER #
        # ========= #
        scheduler = get_scheduler(optimizer=optimizer,
                                  parser=parser)

        # ========= #
        # CRITERION #
        # ========= #
        criterion = get_loss(loss_name=parser.loss,
                             num_label=num_label['all'],
                             parser=parser,
                             device=device)

        # ==================== #
        # INIT METRICS (TRAIN) #
        # ==================== #
        metrics = metrics_dict(metrics_type='train')

        # training epochs range
        start_epoch_train = 1  # star train
        stop_epoch_train = start_epoch_train + parser.epochs  # stop train

        # for each epoch
        for epoch in range(start_epoch_train, stop_epoch_train):

            # ======== #
            # TRAINING #
            # ======== #
            print("\n---------"
                  "\nTRAINING:"
                  "\n---------")
            time_train_start = time.time()
            loss = train(num_epoch=epoch,
                         epochs=parser.epochs,
                         net=net,
                         dataloader=dataloader_train,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         loss_name=parser.loss,
                         criterion=criterion,
                         device=device,
                         parser=parser)
            time_train = time.time() - time_train_start

            # ========== #
            # VALIDATION #
            # ========== #
            print("\n-----------"
                  "\nVALIDATION:"
                  "\n-----------")
            time_val_start = time.time()
            validation(num_epoch=epoch,
                       epochs=parser.epochs,
                       net=net,
                       dataloader=dataloader_val,
                       classifications_path=path['classifications']['validation'],
                       device=device)
            time_val = time.time() - time_val_start

            # ==================== #
            # METRICS (VALIDATION) #
            # ==================== #
            time_metrics_val_start = time.time()

            # ==================== #
            # METRICS (VALIDATION) #
            # ==================== #
            time_metrics_val_start = time.time()

            # read classifications validation for evaluation (numpy array)
            classifications_val = read_csv(filepath_or_buffer=path['classifications']['validation'], usecols=["PREDICTION", "SCORE", "LABEL"])

            # compute accuracy
            accuracy_val = accuracy(classifications=classifications_val)

            # compute MCC
            MCC_val = MCC(classifications=classifications_val)

            # compute ROC AUC
            ROC_AUC_val = ROC_AUC(classifications=classifications_val)

            # compute PR AUC
            PR_AUC_val = PR_AUC(classifications=classifications_val)

            # get current learning rate
            last_learning_rate = current_learning_rate(scheduler=scheduler,
                                                       optimizer=optimizer,
                                                       parser=parser)

            time_metrics_val = time.time() - time_metrics_val_start

            # update performance
            metrics['ticks'].append(epoch)
            metrics['loss'].append(loss)
            metrics['learning_rate'].append(scientific_notation(number=last_learning_rate))
            metrics['accuracy'].append(accuracy_val)
            metrics['MCC'].append(MCC_val)
            metrics['ROC_AUC'].append(ROC_AUC_val)
            metrics['PR_AUC'].append(PR_AUC_val)
            metrics['time']['train'].append(time_train)
            metrics['time']['validation'].append(time_val)
            metrics['time']['metrics'].append(time_metrics_val)

            # metrics-train.csv
            metrics_train_csv(metrics_path=path['metrics']['train'],
                              metrics=metrics)

            # show metrics train
            show_metrics_train(metrics=metrics)

            time_metrics_val = time.time() - time_metrics_val_start

            # =============== #
            # SAVE BEST MODEL #
            # =============== #
            print("\n----------------"
                  "\nSAVE BEST MODEL:"
                  "\n----------------")
            # save best-model with accuracy
            if (epoch - 1) == np.argmax(metrics['accuracy']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['accuracy'],
                                metrics_type='accuracy',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['model']['best']['accuracy'])

            # ========== #
            # PLOT TRAIN #
            # ========== #
            print("\n-----------"
                  "\nPLOT TRAIN:"
                  "\n-----------")
            # figure size
            figsize_x, figsize_y = figure_size(epochs=parser.epochs)

            # epochs ticks
            epochs_ticks = np.arange(1, parser.epochs + 1, step=1)

            # Loss plot
            loss_plot(figsize=(figsize_x, figsize_y),
                      title=plot_title['plots_train']['loss'],
                      experiment_ID=experiment_ID,
                      ticks=metrics['ticks'],
                      epochs_ticks=epochs_ticks,
                      loss=metrics['loss'],
                      loss_path=path['plots_train']['loss'])

            # Accuracy plot
            accuracy_plot(figsize=(figsize_x, figsize_y),
                          title=plot_title['plots_validation']['accuracy'],
                          experiment_ID=experiment_ID,
                          ticks=metrics['ticks'],
                          epochs_ticks=epochs_ticks,
                          accuracy=metrics['accuracy'],
                          accuracy_path=path['plots_validation']['accuracy'])

            # MCC plot
            MCC_plot(figsize=(figsize_x, figsize_y),
                     title=plot_title['plots_validation']['MCC'],
                     experiment_ID=experiment_ID,
                     ticks=metrics['ticks'],
                     epochs_ticks=epochs_ticks,
                     MCC=metrics['MCC'],
                     MCC_path=path['plots_validation']['MCC'])

            # ROC AUC plot
            ROC_AUC_plot(figsize=(figsize_x, figsize_y),
                         title=plot_title['plots_validation']['ROC_AUC'],
                         experiment_ID=experiment_ID,
                         ticks=metrics['ticks'],
                         epochs_ticks=epochs_ticks,
                         ROC_AUC=metrics['ROC_AUC'],
                         accuracy_path=path['plots_validation']['ROC_AUC'])

            # PR AUC plot
            PR_AUC_plot(figsize=(figsize_x, figsize_y),
                        title=plot_title['plots_validation']['PR_AUC'],
                        experiment_ID=experiment_ID,
                        ticks=metrics['ticks'],
                        epochs_ticks=epochs_ticks,
                        PR_AUC=metrics['PR_AUC'],
                        accuracy_path=path['plots_validation']['PR_AUC'])

    # ========== #
    # MODE: TEST #
    # ========== #
    if parser.mode in ['test', 'train_test']:

        # =================== #
        # INIT METRICS (TEST) #
        # =================== #
        metrics = metrics_dict(metrics_type='test')

        # =============== #
        # LOAD BEST MODEL #
        # =============== #
        print("\n----------------"
              "\nLOAD BEST MODEL:"
              "\n----------------")
        # load best model accuracy
        if parser.load_best_accuracy_model:
            load_best_model(net=net,
                            metrics_type='accuracy',
                            path=path['model']['best']['accuracy'])

        # ==== #
        # TEST #
        # ==== #
        print("\n-----"
              "\nTEST:"
              "\n-----")
        time_test_start = time.time()
        test(net=net,
             dataloader=dataloader_test,
             classifications_path=path['classifications']['test'],
             device=device)
        time_test = time.time() - time_test_start

        # ============== #
        # METRICS (TEST) #
        # ============== #
        time_metrics_test_start = time.time()

        # read classifications test for evaluation (numpy array)
        classifications_test = read_csv(filepath_or_buffer=path['classifications']['test'], usecols=["PREDICTION", "SCORE", "LABEL"])

        # compute accuracy
        accuracy_test = accuracy(classifications=classifications_test)

        # compute MCC
        MCC_test = MCC(classifications=classifications_test)

        # compute confusion matrix
        confusion_matrix_test = my_confusion_matrix(classifications=classifications_test)

        # compute precision
        precision_test_micro = precision(classifications=classifications_test,
                                         average='micro')
        precision_test_macro = precision(classifications=classifications_test,
                                         average='macro')
        precision_test_weighted = precision(classifications=classifications_test,
                                            average='weighted')
        precision_test_class = precision_per_class(classifications=classifications_test)

        # compute recall
        recall_test_micro = recall(classifications=classifications_test,
                                   average='micro')
        recall_test_macro = recall(classifications=classifications_test,
                                   average='macro')
        recall_test_weighted = recall(classifications=classifications_test,
                                      average='weighted')
        recall_test_class = recall_per_class(classifications=classifications_test)

        # compute F1-Score
        F1_score_test_micro = F1_score(classifications=classifications_test,
                                       average='micro')
        F1_score_test_macro = F1_score(classifications=classifications_test,
                                       average='macro')
        F1_score_test_weighted = F1_score(classifications=classifications_test,
                                          average='weighted')
        F1_score_test_class = F1_score_per_class(classifications=classifications_test)

        # compute ROC AUC
        ROC_AUC_test = ROC_AUC(classifications=classifications_test)

        # compute PR AUC
        PR_AUC_test = PR_AUC(classifications=classifications_test)

        # compute ROC
        FPR_test, TPR_test = ROC(classifications=classifications_test)

        # compute PR
        precision_test, recall_test = PR(classifications=classifications_test)

        time_metrics_test = time.time() - time_metrics_test_start

        # update performance
        metrics['accuracy'].append(accuracy_test)
        metrics['MCC'].append(MCC_test)
        metrics['precision']['micro'].append(precision_test_micro)
        metrics['precision']['macro'].append(precision_test_macro)
        metrics['precision']['weighted'].append(precision_test_weighted)
        metrics['recall']['micro'].append(recall_test_micro)
        metrics['recall']['macro'].append(recall_test_macro)
        metrics['recall']['weighted'].append(recall_test_weighted)
        metrics['F1']['micro'].append(F1_score_test_micro)
        metrics['F1']['macro'].append(F1_score_test_macro)
        metrics['F1']['weighted'].append(F1_score_test_weighted)
        metrics['class']['precision'] = precision_test_class  # list of precision per class
        metrics['class']['recall'] = recall_test_class  # list of recall per class
        metrics['class']['F1'] = F1_score_test_class  # list of F1-score per class
        metrics['ROC_AUC'].append(ROC_AUC_test)
        metrics['PR_AUC'].append(PR_AUC_test)
        metrics['time']['test'].append(time_test)
        metrics['time']['metrics'].append(time_metrics_test)

        # metrics-test.csv
        metrics_test_csv(metrics_path=path['metrics']['test'],
                         metrics=metrics)

        # confusion-matrix-test.csv
        confusion_matrix_test_csv(classifications=classifications_test,
                                  confusion_matrix=confusion_matrix_test,
                                  confusion_matrix_path=path['metrics']['confusion_matrix_test'],
                                  label_path=path['dataset']['images']['class_label'])

        # metrics-test-class.csv
        metrics_test_class_csv(metrics_class=metrics['class'],
                               label_path=path['dataset']['images']['class_label'],
                               metrics_test_class_path=path['metrics']['test_class'])

        # metrics-test-PR.csv
        metrics_test_PR_csv(metrics_PR_path=path['metrics']['test_PR'],
                            metrics=metrics)

        # show metrics test
        show_metrics_test(metrics=metrics)

        # show metrics test class
        show_metrics_test_class(metrics_class=metrics['class'],
                                label_path=path['dataset']['images']['class_label'])

        # show metrics test PR
        show_metrics_test_PR(metrics=metrics)

        # ====== #
        # OUTPUT #
        # ====== #
        print("\n-------"
              "\nOUTPUT:"
              "\n-------")
        output(dataset=dataset_test,
               num_images=parser.num_images,
               classifications_path=path['classifications']['test'],
               label_path=path['dataset']['images']['class_label'],
               output_path=path['output']['test'],
               suffix="-output|{}".format(experiment_ID))

        # ========= #
        # PLOT TEST #
        # ========= #
        print("\n----------"
              "\nPLOT TEST:"
              "\n----------")
        # ROC plot
        ROC_plot(title=plot_title['plots_test']['ROC'],
                 color='green',
                 experiment_ID=experiment_ID,
                 FPR=FPR_test,
                 TPR=TPR_test,
                 ROC_path=path['plots_test']['ROC'],
                 ROC_coords_path=path['plots_test']['coords']['ROC'])

        # PR plot
        PR_plot(title=plot_title['plots_test']['PR'],
                color='green',
                experiment_ID=experiment_ID,
                precision=precision_test,
                recall=recall_test,
                PR_path=path['plots_test']['PR'],
                PR_coords_path=path['plots_test']['coords']['PR'])

        # Score Distribution
        score_distribution_plot(title=plot_title['plots_test']['score_distribution'],
                                score=classifications_test['SCORE'].values,
                                bins=len(dataset_test),
                                experiment_ID=experiment_ID,
                                score_distribution_path=path['plots_test']['score_distribution'])


if __name__ == "__main__":
    main()
