#!/bin/bash

#  | ==================================== |
#  |      EFFICIENTNET PUV CLASSIFIER     |
#  |                EXPERIMENT            |
#  | ==================================== |

# ---------- #
# PARAMETERS #
# ---------- #
# dataset_path=/data/russo/datasets
# experiments_path=/home/russo/PUV-Project/EfficientNet-PUV-Classifier/experiments
# dataset=SUD4VUP
# split=default
# resize=224
# norm=min-max
# epochs=50
# loss=CE
# optimizer=Adam
# scheduler=ReduceLROnPlateau
# lr=1e-04
# bs=8
# model=EfficientNet-B0
# pretrained
# num_images=63

# CUDA_VISIBLE_DEVICES=2

# ---------- #
# EXPERIMENT #
# ---------- #
# train (default)
CUDA_VISIBLE_DEVICES=2 python3 -u EfficientNet-PUV-Classifier.py train --dataset_path=/data/russo/datasets --experiments_path=/home/russo/PUV-Project/EfficientNet-PUV-Classifier/experiments --dataset=SUD4VUP --split=default --resize=224 --norm=min-max --epochs=50 --loss=CE --optimizer=Adam --scheduler=ReduceLROnPlateau --lr=1e-04 --bs=8 --model=EfficientNet-B0 --pretrained > train.txt

# test (default)
CUDA_VISIBLE_DEVICES=2 python3 -u EfficientNet-PUV-Classifier.py test --dataset_path=/data/russo/datasets --experiments_path=/home/russo/PUV-Project/EfficientNet-PUV-Classifier/experiments --dataset=SUD4VUP --split=default --resize=224 --norm=min-max --epochs=50 --loss=CE --optimizer=Adam --scheduler=ReduceLROnPlateau --lr=1e-04 --bs=8 --model=EfficientNet-B0 --pretrained --num_images=63 > test.txt


