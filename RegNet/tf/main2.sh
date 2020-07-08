#!/bin/bash

SET=$(seq 250 499)
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
    --crop \
    --hflip \
    --jitter 0.1 \
    --lr-mode cosine \
    --checkpoint \
    --history \
    --baseline-path /workspace/nas100/sungchul/Challenge/code_baseline \
    --data-path /workspace/scratch/sungchul/Dataset/imagenet \
    --gpus 2
done