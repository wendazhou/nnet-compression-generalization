#! /usr/bin/env python3.6

""" This file contains the configuration to prune LeNet-5 using Dynamic Network Surgery
on the MNIST dataset.

The original paper claims that we should be able to compress the network (which has 400k parameters)
into 4k non-zero parameters - however some guesswork will be required for the hyperparameters
of the compression as they are not made completely clear.

"""

import sys
import tensorflow as tf
from tensorflow.contrib.training import HParams

from nnet.train_simple import make_model_fn, make_input_fn
from nnet.preprocessing import normalize_preprocessing
from nnet.datasets import mnist_data

from nnet.models.lenet5 import lenet
from functools import partial

from nnet.pruning import CommitMaskedValueHook, masked_variable_getter, make_pruning_summaries, MASKED_WEIGHT_COLLECTION
from nnet.pruning.dns import make_dns_train_op, percentile_thresh_fn


def _trainable_variable_filter(*args, **kwargs):
    return kwargs.get('trainable', True)


def _scope():
    return tf.variable_scope(
        'LeNet',
        custom_getter=masked_variable_getter(mask_filter=_trainable_variable_filter)
    )


def _variables_to_warm_start():
    return [
        v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LeNet')
        if 'Momentum' not in v.name and 'mask' not in v.name
    ]


def _prob_exponential():
    global_step = tf.train.get_or_create_global_step()
    return tf.train.exponential_decay(1.0, global_step, 500, 0.9, name='prob_thresh')


def _prob_inverse():
    global_step = tf.train.get_or_create_global_step()
    return tf.train.inverse_time_decay(
        1.0, global_step, decay_steps=1000, decay_rate=1
    )


def _prob_piecewise():
    global_step = tf.train.get_or_create_global_step()
    return tf.train.piecewise_constant(
        global_step,
        values=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
        boundaries=[500, 4000, 12000, 16000, 20000, 25000]
    )


def _make_thresh_fn(target_iterations, update_steps):
    target_sparsities = {
        'LeNet/conv1/weights': 0.035,
        'LeNet/conv2/weights': 0.03,
        'LeNet/fc1/weights': 0.015,
        'LeNet/fc2/weights': 0.03,
        'LeNet/conv1/bias': 1.00,
        'LeNet/conv2/bias': 0.0,
        'LeNet/fc1/bias': 0.01,
        'LeNet/fc2/bias': 0.0
    }

    def _thresh_fn(variable, mask):
        thresh_fn = percentile_thresh_fn(
            1 - target_sparsities[variable.op.name],
            target_iterations, update_steps,
            thresh_lower_scale=1)

        return thresh_fn(variable, mask)
    
    return _thresh_fn


def _train_op(loss, params):
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)

    weights = tf.get_collection(MASKED_WEIGHT_COLLECTION)

    total_nonzero = tf.add_n([tf.count_nonzero(w) for w in weights])
    total_length = tf.add_n([tf.size(w) for w in weights])

    prune_amount = tf.cast(total_nonzero, tf.float32) / tf.cast(total_length, tf.float32)
    prune_amount = tf.minimum(prune_amount, 0.99)
    prune_penalization = 1 / tf.sqrt((1 - prune_amount))

    regularization_loss = tf.identity(
        prune_penalization * params.l2_pen * tf.add_n([tf.nn.l2_loss(w) for w in weights]),
        name='pruned_regularization_loss')

    loss = loss + regularization_loss
    tf.summary.scalar('pruned_total_loss', loss)

    train_op = make_dns_train_op(
        loss, optimizer=optimizer, thresh_fn_or_scale=_make_thresh_fn(20000, 200),
        prob_thresh=_prob_piecewise(), global_step=global_step)

    return train_op


def _learning_rate():
    global_step = tf.train.get_or_create_global_step()

    return tf.train.piecewise_constant(
        global_step,
        values=[1e-3, 1e-4],
        boundaries=[16000],
        name='learning_rate'
    )


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train', 'evaluate', 'train_and_evaluate'], default='train_and_evaluate')
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--warm-start', type=str)

    args = parser.parse_args()

    params = HParams(
        batch_size=1024,
        image_size=28,
        learning_rate=_learning_rate,
        max_steps=30000,
        l2_pen=0,
        warm_start=args.warm_start)

    model_fn = make_model_fn(partial(lenet, scope=_scope), _train_op)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir, params=params)

    def mnist_input_fn(batch_size, n_epochs=None):
        return make_input_fn(
            mnist_data.get_split(args.dataset_path, 'train'),
            partial(normalize_preprocessing.preprocess, center=True),
            image_size=28,
            batch_size=batch_size,
            n_epochs=n_epochs)

    if args.task in ('train', 'train_and_evaluate'):
        estimator.train(
            mnist_input_fn(params.batch_size),
            max_steps=params.max_steps,
            saving_listeners=[CommitMaskedValueHook(params.max_steps)])
    
    if args.task in ('evaluate', 'train_and_evaluate'):
        estimator.evaluate(mnist_input_fn(1000, 1))


if __name__ == '__main__':
    main()