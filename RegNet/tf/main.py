import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import tqdm
import yaml
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf

from model.anynet import AnyNet
from dataloader import set_dataset
from dataloader import dataloader


def get_argument():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model-name",     type=str,       help='AnyNetXA')
    parser.add_argument("--stamp",          type=int,       default=0)
    parser.add_argument("--dataset",        type=str,       default='imagenet')

    # hyperparameter
    parser.add_argument("--batch-size",     type=int,       default=32, help="batch size per replica")
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.001)
    parser.add_argument("--loss",           type=str,       default='crossentropy')

    parser.add_argument("--augment",        type=str,       default='weak')
    parser.add_argument("--standardize",    type=str,       default='minmax1',      choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])
    parser.add_argument("--pad",            type=int,       default=0,              help='-1: square, 0: no, >1: set')
    parser.add_argument("--crop",           action='store_true')
    parser.add_argument("--angle",          type=int,       default=0)
    parser.add_argument("--vflip",          action='store_true')
    parser.add_argument("--hflip",          action='store_true')
    parser.add_argument("--brightness",     type=float,     default=0.)
    parser.add_argument("--contrast",       type=float,     default=0.)
    parser.add_argument("--saturation",     type=float,     default=0.)
    parser.add_argument("--hue",            type=float,     default=0.)
    parser.add_argument("--jitter",         type=float,     default=0.)
    parser.add_argument("--gray",           action='store_true')
    parser.add_argument("--noise",          type=float,     default=0.)

    # callback
    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--evaluate",       action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr-mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
    parser.add_argument("--lr-value",       type=float,     default=.1)
    parser.add_argument("--lr-interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr-warmup",      type=int,       default=0)

    # etc
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument('--baseline-path',  type=str,       default='/workspace/src/Challenge/code_baseline')
    parser.add_argument('--src-path',       type=str,       default='.')
    parser.add_argument('--data-path',      type=str,       default=None)
    parser.add_argument('--result-path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default=-1)
    parser.add_argument("--ignore-search",  type=str,       default='')

    return parser.parse_args()

def set_cfg(args, logger):
    path = os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp))
    initial_epoch = 0
    
    if os.path.isfile(os.path.join(path, 'history/epoch.csv')):
        df = pd.read_csv(os.path.join(path, 'history/epoch.csv'))
        if len(df) > 0:
            if len(df['epoch'].values) >= args.epochs:
                logger.info('{} Training already finished!!!'.format(args.stamp))
                return args, -1

            else:
                ckpt_list = sorted([d for d in os.listdir(os.path.join(path, 'checkpoint')) if 'h5' in d],
                                key=lambda x: int(x.split('_')[0]))
                print(ckpt_list)
                args.snapshot = os.path.join(path, 'checkpoint/{}'.format(ckpt_list[-1]))
                initial_epoch = int(ckpt_list[-1].split('_')[0])

    desc = yaml.full_load(open(os.path.join(path, 'model_desc.yml'), 'r'))
    for k, v in desc.items():
        if k in ['checkpoint', 'history', 'snapshot', 'gpus', 'src_path', 'data_path', 'result_path']:
            continue
        setattr(args, k, v)

    return args, initial_epoch


def create_model(args, logger):
    if 'anynet' in args.model_name.lower():
        model = AnyNet(args, name='anynet')
    elif 'regnet' in args.model_name.lower():
        pass
    else:
        raise ValueError()

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load model weights at {}'.format(args.snapshot))

    return model

def main():
    args = get_argument()
    assert args.model_name is not None, 'model_name must be set.'

    sys.path.append(args.baseline_path)
    from common import get_logger
    from common import get_session
    from callback_eager import OptionalLearningRateSchedule
    from callback import create_callbacks

    logger = get_logger("MyLogger")

    args, initial_epoch = set_cfg(args, logger)
    if initial_epoch == -1:
        # training was already finished!
        return

    get_session(args)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Strategy
    ##########################
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.CentralStorageStrategy()
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(global_batch_size))

    ##########################
    # Generator
    ##########################
    trainset, valset = set_dataset(args)
    
    train_generator = dataloader(args, trainset, 'train', global_batch_size)
    val_generator = dataloader(args, valset, 'val', global_batch_size, shuffle=False)
    
    steps_per_epoch = args.steps or len(trainset) // global_batch_size
    validation_steps = len(valset) // global_batch_size
    
    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    ##########################
    # Model
    ##########################
    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file=os.path.join(args.src_path, 'model.png'), show_shapes=True)
            model.summary(line_length=130)
            return

        # optimizer
        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.00005)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        else:
            raise ValueError()

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['acc']
        )

    ##########################
    # Callbacks
    ##########################
    callbacks = create_callbacks(
        args, path=os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp)))
    logger.info("Build callbacks!")

    ##########################
    # Train
    ##########################
    model.fit(
        x=train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
        verbose=1,
    )


if __name__ == "__main__":
    main()