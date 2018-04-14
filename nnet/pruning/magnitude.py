""" This module implements magnitude based pruning strategies.

The functionality of this module facilitates implementing strategies
such as LWC as described in https://arxiv.org/pdf/1506.02626.pdf

These are particularly simple strategies based on the magnitude of the
weight, and do not take into account any impact the weight has on the
loss or the output. They usually need to be adapted to in addition
carry out some sort of re-training in order to keep the loss acceptable.

"""

import tensorflow as tf
from . import MASK_COLLECTION


def _make_variable_pruning_op(variable: tf.Variable, threshold, name=None):
    [mask] = tf.get_collection(MASK_COLLECTION, variable.op.name)

    with tf.name_scope(name, default_name='variable_prune_op'):
        to_prune = tf.less_equal(tf.abs(variable), threshold, name='prune_mask')
        remaining = 1 - tf.cast(to_prune, dtype=tf.float32)
        new_mask = tf.multiply(mask, remaining, name='new_mask')
        new_variable = tf.multiply(variable, remaining, name='new_variable')
        assign_mask = mask.assign(new_mask)
        assign_variable = variable.assign(new_variable)

        prune_op = tf.group(assign_mask, assign_variable, name='prune')

    return prune_op


def make_pruning_op(variables, threshold, name=None):
    """ Makes an operation that prunes weights from the given variables that fall
    below a given threshold.

    Parameters
    ----------
    variables: A list of variables to mask. A corresponding mask should
        have been created before hand and exist in the MASK_COLLECTION.
    threshold: A tensor representing the threshold at which to force the
        variables to zero.
    name: A string representing the name of the operation.

    Returns
    -------
    A `tf.Operation`, which when run computes the masks according to the current
    weights of the variables, and updates them.
    """
    with tf.name_scope(name, default_name='prune_op'):
        prune_ops = [_make_variable_pruning_op(variable, threshold) for variable in variables]
        prune_op = tf.group(*prune_ops, name='prune')

    return prune_op


def weights_percentile(variables, percentile, name=None):
    """ Computes the percentile of the entries in the given list of variables.

    Parameters
    ----------
    variables: The list of variables for which to compute the percentile.
    percentile: The percentile (between 0 and 100) to be computed.
    name: An optional name for the operation.

    Returns
    -------
    A `tf.Tensor` representing the percentile of the weights.
    """
    with tf.name_scope(name, default_name='weights_percentile'):
        weights = tf.concat([tf.reshape(v, shape=[-1]) for v in variables], axis=0, name='all_weights')
        weights = tf.abs(weights, name='all_weights_abs')
        percentile = tf.contrib.distributions.percentile(weights, percentile, name='weights_percentile')

    return percentile


def pruning_weight_global_percentile(percentile) -> tf.Operation:
    """ Creates a pruning operation which prunes the weight according to a threshold
    computed from a global percentile.

    Parameters
    ----------
    percentile: The percentile of weights below wish to threshold.

    Returns
    -------
    A `tf.Operation` which thresholds the weights when executed.
    """
    variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
    value = weights_percentile(variables, percentile)

    with tf.control_dependencies([value]):
        return make_pruning_op(variables, value)


def pruning_weight_local_percentile(percentile, variables=None, name=None) -> tf.Operation:
    """ Creates a pruning operation which prunes the weights according to a threshold
    computed from a local percentile.

    Parameters
    ----------
    percentile: The percentile of the weights below wish to threshold.
    variables: The set of variables to prune. Defaults to the WEIGHTS collection.
    name: The name of the operation.

    Returns
    -------
    A `tf.Operation` which thresholds the weights when executed.
    """
    if variables is None:
        variables = tf.get_collection(tf.GraphKeys.WEIGHTS)

    prune_ops = []

    with tf.variable_scope(name, 'pruning_local_percentile'):
        for v in variables:
            with tf.name_scope(v.op.name):
                v_thresh = tf.contrib.distributions.percentile(v, percentile, name='threshold')
                prune = _make_variable_pruning_op(v, v_thresh, name='prune')
            prune_ops.append(prune)

        prune = tf.group(*prune_ops, name='prune')

    return prune
