""" This module contains all the necessary code to perform a compression analysis
on the LeNet-5 model trained on the MNIST dataset.

"""

import pandas as pd

from . import _utils

WEIGHTS_VARIABLES = [
    'LeNet/conv1/weights',
    'LeNet/conv2/weights',
    'LeNet/fc1/weights',
    'LeNet/fc2/weights'
]

BIAS_VARIABLES = [
    'LeNet/conv1/bias',
    'LeNet/conv2/bias',
    'LeNet/fc1/bias',
    'LeNet/fc2/bias'
]


def get_variable_summary(ckpt):
    """ Get a summary of the variables in the given checkpoint.

    This function returns a dataframe with the trainable variables, their shape and length,
    and the number of non-zero elements in the variable.

    Parameters
    ----------
    ckpt: A tensorflow checkpoint reader or a path to a checkpoint.

    Returns
    -------
    variable_summary: a dataframe containing information about the model variables.
    """
    variable_names = WEIGHTS_VARIABLES + BIAS_VARIABLES
    return _utils.get_variable_summary(ckpt, variable_names)


def get_variable_compression_summary(ckpt, scale_posterior=0.1, scale_prior=None,
                                     compression_index=None,
                                     compression_weights=None):
    weights_summary = _utils.get_variable_compression_summary(
        ckpt, WEIGHTS_VARIABLES,
        symmetry_gain=True,
        scale_posterior=scale_posterior,
        scale_prior=scale_prior,
        compression_index=compression_index,
        compression_weights=compression_weights)

    bias_summary = _utils.get_variable_compression_summary(
        ckpt, BIAS_VARIABLES,
        symmetry_gain=False,
        scale_posterior=scale_posterior,
        scale_prior=scale_prior,
        compression_index=compression_index,
        compression_weights=compression_weights)

    return pd.concat((weights_summary, bias_summary))
