""" This module provides the possibility to add random noise to the variables during evaluation.

This helps us evaluate the sensitivity of the weights to noise.

"""

import os
import tensorflow as tf


def _evaluate_function(scale, variable):
    if callable(scale):
        return scale(variable)
    else:
        return scale


def apply_noise(weight, scale, scale_min_max=False, mask=False, name=None):
    """ Returns a noisy version of the given variable.

    This function adds some gaussian noise to the given `weight` tensor. Additionally,
    it implements two important functionality:
    - it can scale the noise depending on the magnitude of the weights (more specifically,
        it looks at the difference between the smallest and largest entry in absolute value.
    - it can add noise to only the non-zero coordinates. This more closely mimics the context
        of adding noise to already pruned variables.

    Parameters
    ----------
    weight: The variable for which to obtain a noisy version.
    scale: The scale of the added noise.
    scale_min_max: Whether to scale the noise according to the magnitude of the entries
        in the random variable.
    mask: if True, only adds noise to non-zero elements of the variable. Otherwise,
        adds noise to every coordinate.
    name: An optional name for the operation.

    Returns
    -------
    A `tf.Tensor` representing the noisy version of the input weight.
    """
    scale = _evaluate_function(scale, weight)

    with tf.name_scope(name, 'NoisyWeight'):
        if scale_min_max:
            variable_abs = tf.abs(weight)
            variable_value = tf.where(
                tf.not_equal(variable_abs, 0.0),
                variable_abs,
                1e6 * tf.ones_like(variable_abs))

            max_bound_init_val = tf.reduce_max(variable_abs, name='max_value')
            min_bound_init_val = tf.minimum(
                tf.reduce_min(variable_value), max_bound_init_val,
                name='min_value')

            scale = tf.cond(
                tf.equal(max_bound_init_val, min_bound_init_val),
                lambda: tf.zeros([], dtype=weight.dtype),
                lambda: scale * (max_bound_init_val - min_bound_init_val)
            )

        noise = tf.random_normal(weight.shape, stddev=scale, name='noise')

        if mask:
            mask_values = tf.cast(tf.not_equal(weight, 0.0), weight.dtype)
            noise = tf.multiply(noise, mask_values, name='masked_noise')

        return tf.add(weight, noise, name='noisy')


def _default_filter(*_, **kwargs):
    collections = kwargs.get('collections', None)
    return collections is not None and tf.GraphKeys.WEIGHTS in collections


def noisy_variable_getter(variable_filter=None, scale=1.0, scale_min_max=False, mask=False):
    """ Create a custom getter which returns noisy versions of the given variables.

    Parameters
    ----------
    variable_filter: A function to filter which variables to add noise to.
    scale: The scale of the noise to be added.
    scale_min_max: If True, additionally scale the noise by the difference between
        the largest and smallest weights in magnitude.
    mask: If True, only add noise to non-zero coordinates.

    Returns
    -------
    custom_getter: The custom getter which returns noisy versions of the variable.
    """
    if variable_filter is None:
        variable_filter = _default_filter

    def custom_getter(getter, *args, **kwargs):
        variable = getter(*args, **kwargs)

        if not variable_filter(*args, **kwargs):
            return variable

        name = os.path.basename(variable.op.name)

        return apply_noise(variable, scale, scale_min_max, mask, name=name + '/noisy')

    return custom_getter
