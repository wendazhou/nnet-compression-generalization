""" Utilities for compression.

"""

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.special import gammaln

from . import coding, bounds


def _ensure_ckpt(ckpt):
    if not hasattr(ckpt, 'get_tensor'):
        ckpt = tf.train.load_checkpoint(ckpt)

    return ckpt


def _get_entropy(counts):
    counts = counts[counts != 0]
    prob = counts / np.sum(counts)
    return -np.vdot(prob, np.log2(prob))


def get_variable_summary(ckpt, variable_names):
    """ Get a summary of the variables in the given checkpoint.

    This function returns a dataframe with the trainable variables, their shape and length,
    and the number of non-zero elements in the variable.

    Parameters
    ----------
    ckpt: A tensorflow checkpoint reader or a path to a checkpoint.
    variable_names: The variables in the checkpoint for which to obtain a summary.

    Returns
    -------
    variable_summary: a dataframe containing information about the model variables.
    """
    ckpt = _ensure_ckpt(ckpt)

    variable_shapes_map = ckpt.get_variable_to_shape_map()

    variable_shapes = [
        variable_shapes_map[v] for v in variable_names
    ]

    variable_length = [
        np.prod(s) for s in variable_shapes
    ]

    variables = [ckpt.get_tensor(v) for v in variable_names]
    variable_idx = [np.flatnonzero(v) for v in variables]

    variable_nonzeros = [np.count_nonzero(v) for v in variables]

    variable_entropy = [
        _get_entropy(np.unique(variable.flat[idx], return_counts=True)[1])
        for variable, idx in zip(variables, variable_idx)
    ]

    idx_entropy = [_get_entropy(np.unique(np.diff(idx), return_counts=True)[1]) for idx in variable_idx]

    dataframe = pd.DataFrame.from_dict({
        'name': variable_names,
        'shape': variable_shapes,
        'num_elem': variable_length,
        'num_nonzero': variable_nonzeros,
        'nonzero_entropy': variable_entropy,
        'idx_entropy': idx_entropy
    })

    dataframe['compression_ratio'] = dataframe['num_nonzero'] / dataframe['num_elem']

    dataframe.set_index('name', inplace=True)

    return dataframe


def _get_smoothing_gains(variable, scale_prior, scale_posterior):
    idx = np.flatnonzero(variable)

    if len(idx) == 0:
        # All values are zero, no noise is added.
        return 0.0, 0.0

    if scale_posterior is None or scale_posterior == 0:
        # Don't add noise to this weight.
        return 0.0, 0.0

    idx_diff = np.ediff1d(idx, to_begin=idx[0] + 1) - 1
    idx_diff = coding.get_index_list(idx_diff)

    idx_all = np.cumsum(np.asarray(idx_diff) + 1) - 1

    if scale_prior is None:
        gain_nats, std_nats = bounds.divergence_gains_opt(
            variable.flat[idx_all],
            scale_posterior,
            n_or_samples=1000)
    else:
        gain_nats, std_nats = bounds.divergence_gains(
            variable.flat[idx_all],
            scale_prior=scale_prior,
            scale_posterior=scale_posterior)

    gain_bytes = -gain_nats / (np.log(2) * 8)
    std_bytes = std_nats / (np.log(2) * 8)

    return gain_bytes, std_bytes


def _call_function(fn_or_value, variable):
    if callable(fn_or_value):
        return fn_or_value(variable)
    else:
        return fn_or_value


def get_variable_compression_summary(
        checkpoint, variable_names, symmetry_gain=True,
        scale_prior=None, scale_posterior=0.1,
        compression_index=None,
        compression_weights=None):
    """ Get a summary of the compression aspect for a single variable.

    Parameters
    ----------
    checkpoint: The checkpoint from which to load the variable.
    variable_names: The names of the variables to load.
    symmetry_gain: Whether to compute the gain from symmetry consideration.
    scale_prior: The scale of the prior distribution.
    scale_posterior: The scale of the posterior distribution.
    compression_index: The type of compression to use for the index.
    compression_weights: The type of compression to use for the weights.

    Returns
    -------
    A dataframe containing the compression information for the given variable.
    """
    checkpoint = _ensure_ckpt(checkpoint)

    shapes = checkpoint.get_variable_to_shape_map()

    coded_lengths = []
    smoothing_gains = []
    smoothing_gains_std = []
    symmetry_gains = []

    for i, variable_name in enumerate(variable_names):
        print('Computing compression for variable {0}'.format(variable_name))

        variable = checkpoint.get_tensor(variable_name)
        coded = coding.compress_variable(
            variable,
            compression_index=_call_function(compression_index, variable),
            compression_weights=_call_function(compression_weights, variable))

        coded_lengths.append(coded.tell())
        smooth_mean, smooth_std = _get_smoothing_gains(
            variable,
            _call_function(scale_prior, variable),
            _call_function(scale_posterior, variable))
        smoothing_gains.append(smooth_mean)
        smoothing_gains_std.append(smooth_std)

        if symmetry_gain and i != len(variable_names) - 1:
            symmetry_gains.append(gammaln(shapes[variable_name][-1] + 1) / (np.log(2) * 8))
        else:
            symmetry_gains.append(0.0)

    return pd.DataFrame.from_dict({
        'name': variable_names,
        'code_length': coded_lengths,
        'smoothing_gain': smoothing_gains,
        'smoothing_gain_std': smoothing_gains_std,
        'symmetry_gain': symmetry_gains
    }).set_index('name')
