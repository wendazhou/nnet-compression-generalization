#! /usr/bin/env python3.6

""" This file contains the configuration to train a baseline LeNet-5 model on the MNIST dataset.

We train a baseline model using mostly standard parameters, except the minibatch size is
set to 1024 for performance reasons, and the learning rate is rescaled to some extent to
account for this. The network trains in a couple of minutes on GPU to a solution with 98+%
accuracy.

"""

import argparse
import sys
import tensorflow as tf

from tensorflow.contrib.training import HParams

from nnet.train_simple import make_input_fn, make_model_fn
from nnet.preprocessing import normalize_preprocessing
from nnet.datasets import mnist_data

from nnet.models.lenet5 import lenet
from functools import partial


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=['train_and_evaluate', 'train', 'evaluate'], default='train_and_evaluate')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--nocenter', action='store_true')

    args = parser.parse_args()

    params = HParams(
        batch_size=1024,
        image_size=28,
        learning_rate = lambda: tf.train.inverse_time_decay(
            learning_rate=0.01,
            decay_rate=1.0,
            decay_steps=50,
            global_step=tf.train.get_or_create_global_step()),
        max_steps=5000,
        l2_pen=0)

    model_fn = make_model_fn(partial(lenet, data_format='channels_first'), None)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir, params=params)

    def mnist_input_fn(batch_size, n_epochs=None):
        return make_input_fn(
            mnist_data.get_split(args.dataset_path, 'train'),
            partial(normalize_preprocessing.preprocess, center=not args.nocenter),
            image_size=28,
            batch_size=batch_size,
            n_epochs=n_epochs)

    if args.task in ('train', 'train_and_evaluate'):
        estimator.train(mnist_input_fn(params.batch_size), max_steps=params.max_steps)

    if args.task in ('evaluate', 'train_and_evaluate'):
        estimator.evaluate(mnist_input_fn(1000, 1))


if __name__ == '__main__':
    main()
