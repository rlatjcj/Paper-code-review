#!/bin/bash

SET=$(seq 0 499)
for stamp in $SET
do
    python main.py \
    --model-name AnyNetXA \
    --stamp $stamp \
    --batch-size 128 \
    --epochs 10 \
    --optimizer sgd \
    --lr 0.05 \
    --standardize norm \
    --lr-mode cosine \
    --data-path /workspace/data2/Dataset/imagenet \
    --gpus 0
done