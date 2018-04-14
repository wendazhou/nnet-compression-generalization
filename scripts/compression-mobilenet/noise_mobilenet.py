#! /bin/env python3.6

""" This script evaluates the stability of a given network to noise
by evaluating the dataset under some added noise to the weights. The
noise is tunable according to some parameters, and is resampled
according to a gaussian at every step.

"""

from functools import partial

import numpy as np
import tensorflow as tf

from nnet.compression.weights_noise import noisy_variable_getter
from nnet.datasets import imagenet_data
from nnet.models.mobilenet import mobilenet_v1
from nnet.preprocessing import inception_preprocessing
from nnet.train_simple import make_input_fn, make_model_fn


def _make_scale(scale_depthwise, scale_pointwise, scale_dense, adj_power=None, cutoff=0.0, depth_multiplier=None):
    def fn(variable):
        if variable.shape[0] == 1:
            if variable.shape[-1] == 1001:
                return scale_dense
            else:
                if adj_power is not None:
                    scale_adj = (
                        int(np.prod(variable.shape)) /
                        max(depth_multiplier * 1024 * 1001, depth_multiplier * depth_multiplier * 1024 * 1024))

                    if cutoff is not None and scale_adj < cutoff:
                        return 0.0

                    return scale_pointwise * np.power(scale_adj, adj_power)
                else:
                    return scale_pointwise
        else:
            return scale_depthwise
    return fn


def _make_scope(scale):
    def _scope():
        return tf.variable_scope(
            'MobilenetV1',
            custom_getter=noisy_variable_getter(
                scale=scale,
                scale_min_max=True,
                mask=True))
    return _scope


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-multiplier', default=0.5, type=float)
    parser.add_argument('--scale-depthwise', default=1.0, type=float)
    parser.add_argument('--scale-pointwise', default=1.0, type=float)
    parser.add_argument('--scale-dense', default=1.0, type=float)
    parser.add_argument('--adj-power', default=None, type=float)
    parser.add_argument('--cutoff', default=None, type=float)
    parser.add_argument('--checkpoint-path', default=None, type=str)
    parser.add_argument('--train-dir', default=None, type=str)
    parser.add_argument('--dataset', default=None, type=str)
    args = parser.parse_args()

    scale = _make_scale(args.scale_depthwise,
                        args.scale_pointwise,
                        args.scale_dense,
                        args.adj_power,
                        args.cutoff,
                        args.depth_multiplier)

    network_fn = partial(mobilenet_v1,
                         num_classes=1001,
                         depth_multiplier=args.depth_multiplier,
                         scope=_make_scope(scale))

    model_fn = make_model_fn(network_fn, None)
    input_fn = make_input_fn(imagenet_data.get_split(args.dataset),
                             inception_preprocessing.preprocess_image,
                             n_epochs=1, image_size=224, batch_size=64)

    estimator = tf.estimator.Estimator(model_fn, model_dir=args.train_dir)
    estimator.evaluate(input_fn, checkpoint_path=args.checkpoint_path)


if __name__ == '__main__':
    main()
