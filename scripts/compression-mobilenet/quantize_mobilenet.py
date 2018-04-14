#! /usr/bin/env python3.6

"""
This file contains configuration to prune MobileNet on the Imagenet dataset.
"""

from functools import partial

import tensorflow as tf

from nnet import quantization
from nnet.datasets import imagenet_data
from nnet.models.mobilenet import mobilenet_v1
from nnet.preprocessing import inception_preprocessing
from nnet.train_simple.hooks import ExecuteAtSessionCreateHook
from nnet.train_simple import make_model_fn, make_input_fn

MINMAX_COLLECTION = 'quantization_minmax'
MINMAX_INIT_COLLECTION = 'minimax_init'


def _train_fn(loss, params):
    global_step = tf.train.get_or_create_global_step()

    variables = tf.get_collection(MINMAX_COLLECTION)
    trainable_variables = set(tf.trainable_variables())

    variables = [v for v in variables if v in trainable_variables]

    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step, variables)

    return train_op


def _is_pointwise_variable(variable):
    shape = variable.shape

    return len(shape) == 4 and shape[0] == 1


def _is_dense(variable):
    shape = variable.shape
    return len(shape) == 4 and shape[-1] == 1001


def _make_num_bits(bits_depthwise, bits_pointwise=None, bits_dense=None):
    if bits_pointwise is None:
        bits_pointwise = bits_depthwise

    if bits_dense is None:
        bits_dense = bits_pointwise

    def num_bits(variable):
        if _is_dense(variable):
            return bits_dense
        elif _is_pointwise_variable(variable):
            return bits_pointwise
        else:
            return bits_depthwise

    return num_bits


def _variable_filter(*_, **kwargs):
    collections = kwargs.get('collections', [])
    collections = [] if collections is None else collections
    return tf.GraphKeys.WEIGHTS in collections


def make_scope(bits_depthwise, bits_pointwise=None, bits_dense=None, passthrough_pointwise=True, use_codebook=False):
    if use_codebook:
        quant = quantization.codebook_quantization(
            _make_num_bits(bits_depthwise, bits_pointwise, bits_dense),
            zero_passthrough=True)
    else:
        quant = quantization.minmax_quantization(
            _make_num_bits(bits_depthwise, bits_pointwise, bits_dense),
            zero_passthrough=_is_pointwise_variable if passthrough_pointwise else False,
            sign_passthrough=_is_pointwise_variable if passthrough_pointwise else False)

    def scope():
        getter = quantization.quantized_variable_getter(
            quant, _variable_filter,
            data_variable_collections=[MINMAX_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES],
            init_collections=[MINMAX_INIT_COLLECTION])

        return tf.variable_scope('MobilenetV1', custom_getter=getter)

    return scope


def get_config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_eval', type=str)
    parser.add_argument('--depth-multiplier', default=0.5, type=float)
    parser.add_argument('--dataset-path', default=None, type=str)
    parser.add_argument('--train-dir', default=None, type=str)
    parser.add_argument('--warm-start', default=None, type=str)
    parser.add_argument('--max-steps', default=20000, type=int)
    parser.add_argument('--num-bits-depthwise', default=8, type=int)
    parser.add_argument('--num-bits-pointwise', default=None, type=int)
    parser.add_argument('--num-bits-dense', default=None, type=int)
    parser.add_argument('--use-codebook', action='store_true')

    args = parser.parse_args()

    network_fn = partial(mobilenet_v1,
                         num_classes=1001,
                         depth_multiplier=args.depth_multiplier,
                         scope=make_scope(args.num_bits_depthwise,
                                          args.num_bits_pointwise,
                                          args.num_bits_dense,
                                          use_codebook=args.use_codebook))

    model_fn = make_model_fn(network_fn, _train_fn)

    return args, model_fn


def _quantize_init_op():
    quantize_init = tf.get_collection(MINMAX_INIT_COLLECTION)
    return tf.group(*quantize_init, name='quantize_init_op')


def train(args, model_fn):
    input_fn = make_input_fn(imagenet_data.get_split(args.dataset_path, shuffle=True),
                             partial(inception_preprocessing.preprocess_image, is_training=False),
                             batch_size=64, image_size=224)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir,
                                       params={'warm_start': args.warm_start})

    estimator.train(
        input_fn,
        hooks=[ExecuteAtSessionCreateHook(op_fn=_quantize_init_op, name='QuantizeInitHook')],
        saving_listeners=[
            quantization.CommitQuantizedValueHook(
                args.max_steps,
                variables_fn=lambda: tf.get_collection(tf.GraphKeys.WEIGHTS)
            )
        ])


def evaluate(args, model_fn):
    input_fn = make_input_fn(imagenet_data.get_split(args.dataset_path, shuffle=False),
                             partial(inception_preprocessing.preprocess_image, is_training=False),
                             batch_size=64, image_size=224)

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
