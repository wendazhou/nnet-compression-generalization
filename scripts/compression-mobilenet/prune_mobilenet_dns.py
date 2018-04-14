#! /usr/bin/env python3.6

"""
This file contains configuration to prune MobileNet on the Imagenet dataset
using the Dynamic Network Surgery strategy.
"""

from functools import partial

import numpy as np
import tensorflow as tf

from nnet.datasets import imagenet_data
from nnet.models.mobilenet import mobilenet_v1
from nnet.preprocessing import inception_preprocessing
from nnet.pruning import masked_variable_getter, CommitMaskedValueHook
from nnet.pruning.dns import make_dns_train_op, percentile_thresh_fn
from nnet.train_simple import make_model_fn, make_input_fn


def _prob_piecewise():
    global_step = tf.train.get_or_create_global_step()
    return tf.train.piecewise_constant(
        global_step,
        values=[1.0, 0.5, 0.25, 0.1, 0.05, 0.05, 0.005],
        boundaries=[5000, 15000, 30000, 50000, 150000, 300000]
    )


def _make_target_sparsity(sparsity_depthwise, sparsity_pointwise, size_offset, depth_multiplier):
    """ Sparsity function that is adaptive depending on whether we are looking at a pointwise
    or depthwise weight. This function also penalizes larger weights (towards the upper layers of
    the network) more than smaller weights (toward the lower layers of the network).
    """
    def fn(variable):
        if variable.shape[0] != 1:
            return sparsity_depthwise
        else:
            size_factor = (int(np.prod(variable.shape))
                           / max(depth_multiplier * 1024 * 1001,
                                 depth_multiplier * depth_multiplier * 1024 * 1024))
            return sparsity_pointwise + size_offset * size_factor

    return fn


def _make_train_op(depth_multiplier=1.0):
    """ Make the DNS train of for the mobilenet network.

    In Mobilenet, the use of depthwise separable convolutions means that the vast
    majority of the computation and weights is spent in the 1x1 pointwise convolutions.

    For this reason, we achieve better performance by focusing the pruning on the pointwise
    convolutions, and only very lightly pruning the depthwise convolutions.

    Parameters
    ----------
    depth_multiplier: The depth multiplier of the network.

    Returns
    -------
    train_op_fn: The function to create the training op.
    """
    def train_op_fn(loss, params):
        prob_thresh = _prob_piecewise()

        optimizer = tf.train.MomentumOptimizer(learning_rate=_learning_rate_inverse, momentum=0.9)
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)

        thresh_fn = percentile_thresh_fn(
            _make_target_sparsity(0.0, 0.60, 0.15, depth_multiplier),
            target_iterations=200000,
            update_steps=2000)

        train_op = make_dns_train_op(
            loss, optimizer=optimizer, prob_thresh=prob_thresh, thresh_fn_or_scale=thresh_fn,
            variables=weights, global_step=tf.train.get_or_create_global_step())

        return train_op

    return train_op_fn


def scope():
    return tf.variable_scope(
        'MobilenetV1',
        custom_getter=masked_variable_getter()
    )


def _learning_rate():
    global_step = tf.train.get_or_create_global_step()

    return tf.cast(tf.train.piecewise_constant(
        global_step,
        values=[0.0001, 0.00005, 0.00001],
        boundaries=[200000, 300000]
    ), tf.float32)


def _learning_rate_inverse():
    global_step = tf.train.get_or_create_global_step()

    return tf.train.inverse_time_decay(
        0.001, global_step, decay_steps=2000, decay_rate=0.05, staircase=True)


def get_config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_eval', type=str)
    parser.add_argument('--depth-multiplier', default=0.5, type=float)
    parser.add_argument('--dataset-path', default=None, type=str)
    parser.add_argument('--max-steps', default=300000, type=int)
    parser.add_argument('--train-dir', default=None, type=str)
    parser.add_argument('--warm-start', default=None, type=str)

    parsed = parser.parse_args()

    network_fn = partial(mobilenet_v1, num_classes=1001,
                         depth_multiplier=parsed.depth_multiplier,
                         scope=scope)

    model_fn = make_model_fn(network_fn, _make_train_op(parsed.depth_multiplier))
    return parsed, model_fn


def train(args, model_fn):
    input_fn = make_input_fn(imagenet_data.get_split(args.dataset_path, shuffle=True),
                             partial(inception_preprocessing.preprocess_image, is_training=False),
                             batch_size=64, image_size=224)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir,
                                       params={'warm_start': args.warm_start})
    estimator.train(input_fn, max_steps=args.max_steps,
                    saving_listeners=[
                        CommitMaskedValueHook(args.max_steps,
                                              variables_fn=lambda: tf.get_collection(tf.GraphKeys.WEIGHTS))
                    ])


def evaluate(args, model_fn):
    input_fn = make_input_fn(imagenet_data.get_split(args.dataset_path, shuffle=False),
                             partial(inception_preprocessing.preprocess_image, is_training=False),
                             image_size=224,
                             n_epochs=1)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir)
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
