#!/usr/bin/env python3.6

"""
This script contains code to analyze the generalization bound obtained from
all the compression procedures on LeNet-5 for the MNISt dataset.
"""

import sys
import argparse
import tensorflow as tf
import numpy as np
from math import log

from nnet.compression.algorithms import AdaptiveArithmeticCompression
from nnet.compression.mnist import get_variable_summary, get_variable_compression_summary
from nnet.compression.bounds import pac_bayes_bound_opt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', default=None, type=str)
    parser.add_argument('--scale-posterior', default=0.1, type=float)
    parser.add_argument('--scale-prior', default=None, type=float)
    parsed = parser.parse_args(sys.argv[1:])

    if parsed.checkpoint_path is None:
        raise ValueError('Must pass checkpoint path to load.')

    ckpt = tf.train.load_checkpoint(parsed.checkpoint_path)
    print_summaries(ckpt, parsed.scale_prior, parsed.scale_posterior)


def print_summaries(ckpt, scale_prior, scale_posterior):
    var_summary = get_variable_summary(ckpt)
    print(var_summary.to_string())

    compression_summary = get_variable_compression_summary(
        ckpt,
        scale_prior=scale_prior,
        scale_posterior=scale_posterior,
        compression_index=AdaptiveArithmeticCompression())
    print(compression_summary.to_string())
    compression_sum = np.maximum(compression_summary, 0).sum()

    code_length = compression_sum['code_length']
    smoothing_gain = compression_sum['smoothing_gain']
    symmetry_gain = compression_sum['symmetry_gain']

    complexity_raw = code_length * 8 * log(2)
    complexity_smoothed = (code_length - smoothing_gain) * 8 * log(2)
    complexity_smooth_sym = (code_length - smoothing_gain - symmetry_gain) * 8 * log(2)

    print('''
Complexity measures (nats)
--------------------------
raw: {0}
smoothed: {1}
smooth and sym: {2}
    '''.format(
        complexity_raw,
        complexity_smoothed,
        complexity_smooth_sym
    ))

    bound_raw = pac_bayes_bound_opt(
        complexity_raw,
        1 - 0.985,
        60000
    )

    bound_smoothed = pac_bayes_bound_opt(
        complexity_smoothed,
        1 - 0.985,
        60000
    )

    bound_both = pac_bayes_bound_opt(
        complexity_smooth_sym,
        1 - 0.985,
        60000
    )

    print('''
Generalization bounds
---------------------
raw: {0}
smoothed: {1}
smooth and sym: {2}'''.format(
        bound_raw,
        bound_smoothed,
        bound_both
    ))


if __name__ == '__main__':
    main()
