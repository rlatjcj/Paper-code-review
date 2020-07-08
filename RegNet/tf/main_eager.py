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

from main import get_argument
from main import set_cfg
from main import create_model
from model.anynet import AnyNet
from dataloader import set_dataset
from dataloader import dataloader
from loss import crossentropy


def main():
    args = get_argument()
    assert args.model_name is not None, 'model_name must be set.'

    sys.path.append(args.baseline_path)
    from common import get_logger
    from common import get_session
    from callback_eager import OptionalLearningRateSchedule
    from callback_eager import create_callbacks

    logger = get_logger("MyLogger")

    args, initial_epoch = set_cfg(args, logger)
    if initial_epoch == -1:
        # training was already finished!
        return

    get_session(args)

    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args)

    ##########################
    # Model & Metric & Generator
    ##########################
    progress_desc_train = 'Train : Loss {:.4f} | Acc {:.4f}'
    progress_desc_val = 'Val : Loss {:.4f} | Acc {:.4f}'

    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    steps_per_epoch = args.steps or len(trainset) // global_batch_size
    validation_steps = len(valset) // global_batch_size

    # lr scheduler
    lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)

    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            from tensorflow.keras.utils import plot_model
            plot_model(model, to_file=os.path.join(args.src_path, 'model.png'), show_shapes=True)
            model.summary(line_length=130)
            return

        # metrics
        metrics = {
            'loss'      :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'acc'       :   tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32),
            'val_loss'  :   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
            'val_acc'   :   tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32),
        }

        # optimizer
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.00005)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        else:
            raise ValueError()

        # loss
        if args.loss == 'crossentropy':
            criterion = crossentropy(args)
        else:
            raise ValueError()

        # generator
        if args.loss == 'crossentropy':
            train_generator = dataloader(args, trainset, 'train', global_batch_size)
            val_generator = dataloader(args, valset, 'val', global_batch_size, shuffle=False)
        else:
            raise ValueError()

        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    path = os.path.join(args.result_path, args.dataset, args.model_name, str(args.stamp))
    csvlogger, train_writer, val_writer = create_callbacks(args, metrics, path)
    logger.info("Build Model & Metrics")

    ##########################
    # Log Arguments & Settings
    ##########################
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    logger.info("GLOBAL BATCH SIZE : {}".format(global_batch_size))

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
        
    @tf.function
    def do_step(iterator, mode, loss_name, acc_name=None):
        def step_fn(from_iterator):
            inputs, labels = from_iterator
            if mode == 'train':
                # TODO : loss 계산 다시하기
                with tf.GradientTape() as tape:
                    logits = tf.cast(model(inputs, training=True), tf.float32)
                    loss = criterion(labels, logits)
                    loss = tf.reduce_sum(loss) * (1./global_batch_size)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            else:
                logits = tf.cast(model(inputs, training=False), tf.float32)
                loss = criterion(labels, logits)
                loss = tf.reduce_sum(loss) * (1./global_batch_size)

            metrics[loss_name].update_state(loss)
            metrics[acc_name].update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        # step_fn(next(iterator))

    def desc_update(pbar, desc, loss, acc=None):
        pbar.set_description(desc.format(loss.result(), acc.result()))


    ##########################
    # Train
    ##########################
    for epoch in range(initial_epoch, args.epochs):
        print('\nEpoch {}/{}'.format(epoch+1, args.epochs))
        print('Learning Rate : {}'.format(optimizer.learning_rate(optimizer.iterations)))

        # train
        progressbar_train = tqdm.tqdm(
            tf.range(steps_per_epoch), 
            desc=progress_desc_train.format(0, 0, 0, 0), 
            leave=True)
        for step in progressbar_train:
            do_step(train_iterator, 'train', 'loss', 'acc')
            desc_update(progressbar_train, progress_desc_train, metrics['loss'], metrics['acc'])
            progressbar_train.refresh()

        # eval
        progressbar_val = tqdm.tqdm(
            tf.range(validation_steps), 
            desc=progress_desc_val.format(0, 0), 
            leave=True)
        for step in progressbar_val:
            do_step(val_iterator, 'val', 'val_loss', 'val_acc')
            desc_update(progressbar_val, progress_desc_val, metrics['val_loss'], metrics['val_acc'])
            progressbar_val.refresh()
    
        # logs
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch

        if args.checkpoint:
            ckpt_path = '{:04d}_{:.4f}_{:.4f}.h5'.format(epoch+1, logs['val_acc'], logs['val_loss'])
            model.save_weights(os.path.join(path, 'checkpoint', ckpt_path))
            print('\nSaved at {}'.format(os.path.join(path, 'checkpoint', ckpt_path)))

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(path, 'history/epoch.csv'), index=False)

        if args.tensorboard:
            with train_writer.as_default():
                tf.summary.scalar('loss', metrics['loss'].result(), step=epoch)
                tf.summary.scalar('acc', metrics['acc'].result(), step=epoch)

            with val_writer.as_default():
                tf.summary.scalar('val_loss', metrics['val_loss'].result(), step=epoch)
                tf.summary.scalar('val_acc', metrics['val_acc'].result(), step=epoch)
        
        for k, v in metrics.items():
            v.reset_states()


if __name__ == "__main__":
    main()