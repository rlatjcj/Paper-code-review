import os
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append('/workspace/src/code_baseline')
from generator.augment import SimAugment


AUTO = tf.data.experimental.AUTOTUNE

mean_std = {
    'cub': [[0.48552202, 0.49934904, 0.43224954], 
            [0.18172876, 0.18109447, 0.19272076]],
    'cifar100': [[0.50707516, 0.48654887, 0.44091784], 
                 [0.20079844, 0.19834627, 0.20219835]],
}

def set_dataset(args):
    trainset = pd.read_csv(
        os.path.join(
            args.data_path, '{}_trainset.csv'.format(args.dataset)
        )).values.tolist()
    valset = pd.read_csv(
        os.path.join(
            args.data_path, '{}_valset.csv'.format(args.dataset)
        )).values.tolist()
    return np.array(trainset, dtype='object'), np.array(valset, dtype='object')

#############################################################################
def fetch_dataset(path, y):
    x = tf.io.read_file(path)
    return tf.data.Dataset.from_tensors((x, y))

def dataloader(args, datalist, mode, shuffle=True):
    '''dataloader for cross-entropy loss
    '''
    def augmentation(img, label, shape):
        if args.augment == 'sim':
            augment = SimAugment(args, mode)

        for f in augment.augment_list:
            if 'crop' in f.__name__:
                img = f(img, shape)
            else:
                img = f(img)
        
        # one-hot encodding
        label = tf.one_hot(label, args.classes)
        return img, label

    def preprocess_image(img, label):
        shape = tf.image.extract_jpeg_shape(img)
        img = tf.io.decode_jpeg(img, channels=3)
        img, label = augmentation(img, label, shape)
        return (img, label)

    imglist, labellist = datalist[:,0].tolist(), datalist[:,1].tolist()
    imglist = [os.path.join(args.data_path, i) for i in imglist]

    dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(datalist))

    dataset = dataset.interleave(fetch_dataset, num_parallel_calls=AUTO)
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset


def dataloader_supcon(args, datalist, mode, dtype=None, shuffle=True):
    '''dataloader for supervised contrastive loss
    '''
    def augmentation(img, shape):
        if args.augment == 'sim':
            augment = SimAugment(args, mode)
            
        aug_img = tf.identity(img)
        for f in augment.augment_list:
            if 'crop' in f.__name__:
                aug_img = f(aug_img, shape)
            else:
                aug_img = f(aug_img)
        
        return aug_img, aug_img

    def preprocess_image(img, label):
        shape = tf.image.extract_jpeg_shape(img)
        img = tf.io.decode_jpeg(img, channels=3)
        anchor, aug_img = augmentation(img, shape)
        return (anchor, aug_img), [label]

    imglist, labellist = datalist[:,0].tolist(), datalist[:,1].tolist()
    imglist = [os.path.join(args.data_path, i) for i in imglist]

    dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(datalist))

    dataset = dataset.interleave(fetch_dataset, num_parallel_calls=AUTO)
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset