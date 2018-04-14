#! /usr/bin/env python3

""" This file contains a script to prune ResNet-56 on the CIFAR-10 dataset, and its randomized versions.
"""

import tensorflow as tf

from nnet.datasets import cifar_random
from nnet.models.resnet_model import cifar10_resnet_v2_generator
from nnet.preprocessing import normalize_preprocessing
from nnet.pruning import masked_variable_getter, CommitMaskedValueHook
from nnet.pruning.dns import percentile_thresh_fn, make_dns_train_op
from nnet.train_simple import make_input_fn, make_model_fn


def _learning_rate():
    return tf.train.inverse_time_decay(
        0.05, tf.train.get_or_create_global_step(),
        decay_steps=200,
        decay_rate=0.1)


def _make_train_op(target_sparsity):
    """ Make the training op to prune up to the given target sparsity.

    Parameters
    ----------
    target_sparsity: The desired final sparsity of the network.

    Returns
    -------
    train_op_fn: A function which creates the train op.
    """
    def train_op_fn(loss, params):
        prob_thresh = tf.train.inverse_time_decay(
            1.0, tf.train.get_or_create_global_step(),
            decay_steps=100, decay_rate=0.2,
            staircase=True, name='prob_thresh')

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=_learning_rate, momentum=0.9)

        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)

        thresh_fn = percentile_thresh_fn(
            target_sparsity,
            target_iterations=15000,
            update_steps=500)

        train_op = make_dns_train_op(
            loss, optimizer=optimizer, prob_thresh=prob_thresh, thresh_fn_or_scale=thresh_fn,
            variables=weights, global_step=tf.train.get_or_create_global_step())

        return train_op

    return train_op_fn


def _scope():
    return tf.variable_scope('', custom_getter=masked_variable_getter())


def get_config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_eval', type=str)
    parser.add_argument('--target-sparsity', default=0.5, type=float)
    parser.add_argument('--proportion-random', default=0.0, type=float)
    parser.add_argument('--max-steps', default=20000, type=int)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--warm-start', type=str, default=None)
    args = parser.parse_args()

    network_fn = cifar10_resnet_v2_generator(56, 10, model_scope=_scope)
    model_fn = make_model_fn(network_fn, _make_train_op(args.target_sparsity))

    return args, model_fn


def train(args, model_fn):
    input_fn = make_input_fn(cifar_random.get_split(args.dataset, proportion_random=args.proportion_random),
                             preprocessing_fn=normalize_preprocessing.preprocess,
                             image_size=32, batch_size=128)

    estimator = tf.estimator.Estimator(
        model_fn, model_dir=args.model_dir,
        params={'warm_start': args.warm_start})

    estimator.train(
        input_fn,
        saving_listeners=[
            CommitMaskedValueHook(args.max_steps,
                                  variables_fn=lambda: tf.get_collection(tf.GraphKeys.WEIGHTS))
        ])


def evaluate(args, model_fn):
    input_fn = make_input_fn(cifar_random.get_split(args.dataset, proportion_random=args.proportion_random),
                             preprocessing_fn=normalize_preprocessing.preprocess,
                             image_size=32, batch_size=128, n_epochs=1)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.model_dir)
    estimator.evaluate(input_fn)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args, model_fn = get_config()

    if args.train_or_eval == 'train':
        train(args, model_fn)
    elif args.train_or_eval == 'eval':
        evaluate(args, model_fn)
    else:
        raise ValueError('Must pass train or eval')


if __name__ == '__main__':
    main()
