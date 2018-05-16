#! /usr/bin/env python3.6

"""
This file contains configuration to quantize LeNet-5 on the Mnist dataset.
"""

import tensorflow as tf
from tensorflow.contrib.training import HParams

from nnet.train_simple import make_input_fn, make_model_fn
from nnet.train_simple.hooks import ExecuteAtSessionCreateHook
from nnet.preprocessing import normalize_preprocessing
from nnet.datasets import mnist_data
from nnet.quantization import quantized_variable_getter, minmax_quantization, codebook_quantization, CommitQuantizedValueHook

from nnet.models.lenet5 import lenet
from functools import partial

MINMAX_COLLECTION = 'quantization_minmax'
MINMAX_INIT_COLLECTION = 'minimax_init'


def _train_fn(loss, params):
    global_step = tf.train.get_or_create_global_step()

    trainable_variables = set(tf.trainable_variables())

    variables = [v for v in tf.get_collection(MINMAX_COLLECTION) if v in trainable_variables]

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step, variables)

    return train_op


def _minmax_init_op():
    minmax_init = tf.get_collection(MINMAX_INIT_COLLECTION)
    return tf.group(*minmax_init, name='minmax_init_op')


def _variable_filter(*_, **kwargs):
    collections = kwargs.get('collections', None)
    return collections is not None and (tf.GraphKeys.WEIGHTS in collections or tf.GraphKeys.BIASES in collections)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', choices=['train', 'evaluate', 'train_and_evaluate'], default='train_and_evaluate')
    parser.add_argument('--dataset-path', type=str, help='the directory containing the MNIST data')
    parser.add_argument('--warm-start', type=str, help='a checkpoint from which to warm start')
    parser.add_argument('--num-bits', type=int, default=4, help='the number of bits to use in the codebook')
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--max-steps', default=2000, type=int)

    args = parser.parse_args()

    params = HParams(
        batch_size=1024,
        max_steps=args.max_steps,
        warm_start=args.warm_start)

    def scope():
        codebook_quant = codebook_quantization(
            num_bits=args.num_bits,
            zero_passthrough=True)

        return tf.variable_scope(
            'LeNet',
            custom_getter=quantized_variable_getter(
                codebook_quant,
                _variable_filter,
                data_variable_collections=[MINMAX_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES],
                init_collections=[MINMAX_INIT_COLLECTION]
            ))
    
    def mnist_input_fn(batch_size, n_epochs=None):
        return make_input_fn(
            mnist_data.get_split(args.dataset_path, 'train'),
            partial(normalize_preprocessing.preprocess),
            image_size=28,
            batch_size=batch_size,
            n_epochs=n_epochs)
    
    model_fn = make_model_fn(partial(lenet, scope=scope), _train_fn)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir)

    if args.task in ('train', 'train_and_evaluate'):
        estimator.train(
            mnist_input_fn(params.batch_size),
            max_steps=params.max_steps,
            hooks=[ExecuteAtSessionCreateHook(op_fn=_minmax_init_op, name='MinmaxInitHook')],
            saving_listeners=[CommitQuantizedValueHook(params.max_steps)])

    if args.task in ('evaluate', 'train_and_evaluate'):
        estimator.evaluate(mnist_input_fn(1000, 1))


if __name__ == '__main__':
    main()
