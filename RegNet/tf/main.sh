#!/bin/bash

SET=$(seq 0 499)
for stamp in $SET
do
    python main.py \
    --model-name AnyNetXA \
    --stamp $stamp \
    --dataset cifar100 \
    --batch-size 128 \
    --epochs 10 \
    --optimizer sgd \
    --lr 0.05 \
    --standardize norm \
    --pad 4 \
    --crop \
    --hflip \
    --jitter 0.1 \
    --lr-mode cosine \
    --checkpoint \
    --history \
    --data-path /workspace/data2/Dataset/cifar100 \
    --gpus 3
done