#! /usr/bin/env python3.6

""" This file contains code to evaluate the network when noise is added to the weights.

"""

import sys
import argparse
import tensorflow as tf

from nnet.train_simple import make_input_fn, make_model_fn
from nnet.preprocessing import normalize_preprocessing
from nnet.datasets import mnist_data

from nnet.compression.weights_noise import noisy_variable_getter

from nnet.models.lenet5 import lenet
from functools import partial


def _make_scope(scale):
    def _scope():
        return tf.variable_scope(
            'LeNet',
            custom_getter=noisy_variable_getter(scale=scale, scale_min_max=True, mask=True))
    return _scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--split', default='train')
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--nocenter', action='store_true')

    args = parser.parse_args()

    input_fn = make_input_fn(
        mnist_data.get_split(args.dataset_path, args.split),
        partial(normalize_preprocessing.preprocess, center=not args.nocenter),
        image_size=28,
        shuffle=False,
        batch_size=1000,
        n_epochs=1)
    
    model_fn = make_model_fn(
        partial(lenet, scope=_make_scope(args.scale)), None)
    
    estimator = tf.estimator.Estimator(model_fn)
    metrics = estimator.evaluate(input_fn, checkpoint_path=args.checkpoint_path)
    print('Finished evaluation. Obtained metrics: {0}'.format(metrics))

if __name__ == '__main__':
    main()
