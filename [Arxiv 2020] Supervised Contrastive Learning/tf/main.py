import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import tqdm
import argparse
import numpy as np
from datetime import datetime

from dataloader import set_dataset
from dataloader import dataloader
from dataloader import dataloader_supcon
from loss import crossentropy
from loss import supervised_contrastive

import tensorflow as tf


model_dict = {
    'vgg16'         : tf.keras.applications.VGG16,
    'vgg19'         : tf.keras.applications.VGG19,
    'resnet50'      : tf.keras.applications.ResNet50,
    'resnet50v2'    : tf.keras.applications.ResNet50V2,
    'resnet101'     : tf.keras.applications.ResNet101,
    'resnet101v2'   : tf.keras.applications.ResNet101V2,
    'resnet152'     : tf.keras.applications.ResNet152,
    'resnet152v2'   : tf.keras.applications.ResNet152V2,
    'xception'      : tf.keras.applications.Xception, # 299
    'densenet121'   : tf.keras.applications.DenseNet121, # 224
    'densenet169'   : tf.keras.applications.DenseNet169, # 224
    'densenet201'   : tf.keras.applications.DenseNet201, # 224
}

def create_model(args, logger):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.models import Model

    backbone = model_dict[args.backbone](
        include_top=False,
        pooling='avg',
        weights=None,
        input_shape=(args.img_size, args.img_size, 3))

    if args.loss == 'crossentropy':
        x = Dense(args.classes)(backbone.output)
        x = Activation('softmax', name='main_output')(x)
    elif args.loss == 'supcon':
        x = Dense(2048, name='proj_hidden')(backbone.output)
        x = Dense(128, name='proj_output')(x)
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='main_output')(x)
    model = Model(backbone.input, x, name=args.backbone)

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load weights at {}'.format(args.snapshot))
    return model


def main(args):
    sys.path.append(args.baseline_path)
    from common import get_logger
    from common import get_session
    from common import search_same
    from callback_eager import OptionalLearningRateSchedule
    from callback_eager import create_callbacks

    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        temp = datetime.now()
        args.stamp = "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
            temp.year // 100,
            temp.month,
            temp.day,
            weekday[temp.weekday()],
            temp.hour,
            temp.minute,
            temp.second,
        )

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args)

    ##########################
    # Model & Metric & Generator
    ##########################
    progress_desc_train = 'Train : Loss {:.4f}'
    progress_desc_val = 'Val : Loss {:.4f}'
    if args.loss == 'crossentropy':
        progress_desc_train += ' | Acc {:.4f}'
        progress_desc_val += ' | Acc {:.4f}'

    # select your favorite distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
    logger.info('{} : {}'.format(strategy.__class__.__name__, strategy.num_replicas_in_sync))
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    logger.info("GLOBAL BATCH SIZE : {}".format(global_batch_size))

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== trainset ==========")
    steps_per_epoch = args.steps or len(trainset) // global_batch_size
    logger.info("    --> {}".format(len(trainset)))
    logger.info("    --> {}".format(steps_per_epoch))

    logger.info("=========== valset ===========")
    validation_steps = len(valset) // global_batch_size
    logger.info("    --> {}".format(len(valset)))
    logger.info("    --> {}".format(validation_steps))

    # lr scheduler
    lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)

    with strategy.scope():
        model = create_model(args, logger)
        if args.summary:
            model.summary()
            return

        # metrics
        metrics = {
            'loss'    :   tf.keras.metrics.Mean('loss', dtype=tf.float32),
            'val_loss':   tf.keras.metrics.Mean('val_loss', dtype=tf.float32),
        }

        # optimizer
        if args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=.9, decay=.0001)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
        elif args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # loss
        if args.loss == 'supcon':
            criterion = supervised_contrastive(args)
        else:
            criterion = crossentropy(args)
            metrics['acc'] = tf.keras.metrics.CategoricalAccuracy('acc', dtype=tf.float32)
            metrics['val_acc'] = tf.keras.metrics.CategoricalAccuracy('val_acc', dtype=tf.float32)

        # generator
        if args.loss == 'crossentropy':
            train_generator = dataloader(args, trainset, 'train', global_batch_size)
            val_generator = dataloader(args, valset, 'val', global_batch_size, shuffle=False)
        elif args.loss =='supcon':
            train_generator = dataloader_supcon(args, trainset, 'train', global_batch_size)
            val_generator = dataloader_supcon(args, valset, 'train', global_batch_size, shuffle=False)
        else:
            raise ValueError()
        
        train_generator = strategy.experimental_distribute_dataset(train_generator)
        val_generator = strategy.experimental_distribute_dataset(val_generator)

    csvlogger = create_callbacks(args, metrics)
    logger.info("Build Model & Metrics")

    ##########################
    # READY Train
    ##########################
    train_iterator = iter(train_generator)
    val_iterator = iter(val_generator)
        
    @tf.function
    def do_step(iterator, mode, loss_name, acc_name=None):
        def step_fn(from_iterator):
            if args.loss == 'supcon':
                (img1, img2), labels = from_iterator
                inputs = tf.concat([img1, img2], axis=0)
            else:
                inputs, labels = from_iterator

            if mode == 'train':
                with tf.GradientTape() as tape:
                    logits = tf.cast(model(inputs, training=True), tf.float32)
                    loss = criterion(labels, logits)
                    loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            else:
                logits = tf.cast(model(inputs, training=False), tf.float32)
                loss = criterion(labels, logits)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)

            metrics[loss_name].update_state(loss * strategy.num_replicas_in_sync)
            # metrics[loss_name].update_state(loss)
            if args.loss == 'crossentropy':
                metrics[acc_name].update_state(labels, logits)

        strategy.run(step_fn, args=(next(iterator),))
        # step_fn(next(iterator))

    def desc_update(pbar, desc, loss, acc=None):
        if args.loss == 'supcon':
            pbar.set_description(desc.format(loss.result()))
        else:
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
            # train_step(train_iterator)
            desc_update(progressbar_train, progress_desc_train, metrics['loss'], 
                        None if args.loss == 'supcon' else metrics['acc'])
            progressbar_train.refresh()

        # eval
        progressbar_val = tqdm.tqdm(
            tf.range(validation_steps), 
            desc=progress_desc_val.format(0, 0), 
            leave=True)
        for step in progressbar_val:
            do_step(val_iterator, 'val', 'val_loss', 'val_acc')
            # eval_step(val_iterator)
            desc_update(progressbar_val, progress_desc_val, metrics['val_loss'], 
                        None if args.loss == 'supcon' else metrics['val_acc'])
            progressbar_val.refresh()
    
        # logs
        logs = {k: v.result().numpy() for k, v in metrics.items()}
        logs['epoch'] = epoch + 1

        if args.checkpoint:
            if args.loss == 'supcon':
                ckpt_path = '{:04d}_{:.4f}.h5'.format(epoch+1, logs['val_loss'])
            else:
                ckpt_path = '{:04d}_{:.4f}_{:.4f}.h5'.format(epoch+1, logs['val_acc'], logs['val_loss'])

            model.save_weights(
                os.path.join(
                    args.result_path, 
                    '{}/{}/checkpoint'.format(args.dataset, args.stamp),
                    ckpt_path))

            print('\nSaved at {}'.format(
                os.path.join(
                    args.result_path, 
                    '{}/{}/checkpoint'.format(args.dataset, args.stamp),
                    ckpt_path)))

        if args.history:
            csvlogger = csvlogger.append(logs, ignore_index=True)
            csvlogger.to_csv(os.path.join(args.result_path, '{}/{}/history/epoch.csv'.format(args.dataset, args.stamp)), index=False)
        
        for k, v in metrics.items():
            v.reset_states()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",       type=str,       default='resnet50')
    parser.add_argument("--batch-size",     type=int,       default=32,
                        help="batch size per replica")
    parser.add_argument("--classes",        type=int,       default=200)
    parser.add_argument("--dataset",        type=str,       default='cub')
    parser.add_argument("--img-size",       type=int,       default=224)
    parser.add_argument("--steps",          type=int,       default=0)
    parser.add_argument("--epochs",         type=int,       default=100)

    parser.add_argument("--optimizer",      type=str,       default='sgd')
    parser.add_argument("--lr",             type=float,     default=.001)
    parser.add_argument("--loss",           type=str,       default='crossentropy', choices=['crossentropy', 'supcon'])
    parser.add_argument("--temperature",    type=float,     default=0.007)

    parser.add_argument("--augment",        type=str,       default='sim')
    parser.add_argument("--standardize",    type=str,       default='minmax1',      choices=['minmax1', 'minmax2', 'norm', 'eachnorm'])

    parser.add_argument("--checkpoint",     action='store_true')
    parser.add_argument("--history",        action='store_true')
    parser.add_argument("--tensorboard",    action='store_true')
    parser.add_argument("--lr-mode",        type=str,       default='constant',     choices=['constant', 'exponential', 'cosine'])
    parser.add_argument("--lr-value",       type=float,     default=.1)
    parser.add_argument("--lr-interval",    type=str,       default='20,50,80')
    parser.add_argument("--lr-warmup",      type=int,       default=0)

    parser.add_argument('--baseline-path',  type=str,       default='/workspace/src/Challenge/code_baseline')
    parser.add_argument('--src-path',       type=str,       default='.')
    parser.add_argument('--data-path',      type=str,       default=None)
    parser.add_argument('--result-path',    type=str,       default='./result')
    parser.add_argument('--snapshot',       type=str,       default=None)
    parser.add_argument("--gpus",           type=str,       default=-1)
    parser.add_argument("--summary",        action='store_true')
    parser.add_argument("--ignore-search",  type=str,       default='')

    main(parser.parse_args())