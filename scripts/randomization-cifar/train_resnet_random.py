#! /usr/bin/env python3.6

import tensorflow as tf

from nnet.preprocessing import normalize_preprocessing
from nnet.datasets import cifar_random
from nnet.train_simple import make_input_fn, make_model_fn

from nnet.models.resnet_model import cifar10_resnet_v2_generator


def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_eval', type=str)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--train-dir', default=None, type=str)
    parser.add_argument('--model-size', default=56, type=int)
    parser.add_argument('--proportion-random', default=1.0, type=float)
    args = parser.parse_args()

    network_fn = cifar10_resnet_v2_generator(args.model_size, 10)
    model_fn = make_model_fn(network_fn, None)

    return args, model_fn


def train(args, model_fn):
    input_fn = make_input_fn(
        cifar_random.get_split(args.dataset, num_classes=10, proportion_random=args.proportion_random),
        preprocessing_fn=normalize_preprocessing.preprocess,
        batch_size=128, image_size=32)

    params = {
        'learning_rate': lambda: tf.train.piecewise_constant(
            tf.train.get_global_step(),
            [tf.constant(x, dtype=tf.int64) for x in [400, 32000, 48000]],
            [0.001, 0.1, 0.01, 0.001])
    }

    estimator = tf.estimator.Estimator(model_fn, args.train_dir, params=params)
    estimator.train(input_fn, max_steps=64000)


def evaluate(args, model_fn):
    input_fn = make_input_fn(
        cifar_random.get_split(args.dataset, num_classes=10, proportion_random=args.proportion_random),
        preprocessing_fn=normalize_preprocessing.preprocess,
        batch_size=128, image_size=32, n_epochs=1, shuffle=False)

    estimator = tf.estimator.Estimator(model_fn, args.train_dir)
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

