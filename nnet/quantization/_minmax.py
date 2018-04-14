""" Simple min-max implementation of quantization.

This module implements a simple linear interpolation strategy
for quantization.

"""

import tensorflow as tf


def _evaluate_fn(fn, variable):
    if callable(fn):
        return fn(variable)
    else:
        return fn


def make_minmax_config(variable, num_bits, zero_passthrough, sign_passthrough):
    num_bits = _evaluate_fn(num_bits, variable)
    zero_passthrough = _evaluate_fn(zero_passthrough, variable)
    sign_passthrough = _evaluate_fn(sign_passthrough, variable)

    if sign_passthrough and not zero_passthrough:
        raise ValueError('sign_passthrough is only allowed with zero_passthrough.')

    return {
        'variable': variable,
        'num_bits': num_bits,
        'zero_passthrough': zero_passthrough,
        'sign_passthrough': sign_passthrough,
    }


def make_minmax_data(config, data_collections):
    sign_passthrough = config['sign_passthrough']

    min_bound = tf.get_variable(
        'min', dtype=tf.float32, trainable=True,
        collections=data_collections,
        initializer=-6.0,
        constraint=lambda v: tf.maximum(v, 0.0, name='constraint_pos') if sign_passthrough else None)

    max_bound = tf.get_variable(
        'max', dtype=tf.float32, trainable=True, initializer=6.0,
        collections=data_collections)

    return {'min_bound': min_bound, 'max_bound': max_bound}


def make_minmax_quantization(config, data):
    variable = config['variable']
    sign_passthrough = config['sign_passthrough']
    num_bits = config['num_bits']
    zero_passthrough = config['zero_passthrough']

    min_bound = data['min_bound']
    max_bound = data['max_bound']

    quantized_variable = tf.fake_quant_with_min_max_vars(
        tf.abs(variable) if sign_passthrough else variable,
        min_bound, max_bound,
        num_bits=num_bits - int(sign_passthrough),
        narrow_range=zero_passthrough,
        name='quantized')

    if sign_passthrough:
        # this implies zero_passthrough
        sign = tf.sign(variable, name='var_sign')
        quantized_variable = tf.multiply(
            quantized_variable, sign, name='quantized_with_sign')
    elif zero_passthrough:
        mask = tf.not_equal(variable, 0.0, name='sparse_mask')
        quantized_variable = tf.multiply(
            quantized_variable,
            tf.cast(mask, quantized_variable.dtype),
            name='quantized_masked')

    return quantized_variable


def make_minmax_init(config, data):
    variable = config['variable']
    sign_passthrough = config['sign_passthrough']

    min_bound = data['min_bound']
    max_bound = data['max_bound']

    if isinstance(variable, tf.Variable):
        variable_value = variable.initialized_value()
    else:
        variable_value = variable

    if sign_passthrough:
        variable_value = tf.abs(variable_value, name='var_value_abs')

    max_bound_init_val = tf.reduce_max(variable_value, name='max_value')

    if sign_passthrough:
        # We must take the minimum of the non-zero values, except if
        # all values are zero in which case we put min = max = 0.
        variable_value = tf.where(
            tf.not_equal(variable_value, 0.0),
            variable_value,
            1e6 * tf.ones_like(variable_value))
        min_bound_init_val = tf.minimum(
            tf.reduce_min(variable_value), max_bound_init_val,
            name='min_value')
    else:
        min_bound_init_val = tf.reduce_min(variable_value, name='min_value')

    init_min = tf.assign(min_bound, min_bound_init_val, name='init_min')
    init_max = tf.assign(max_bound, max_bound_init_val, name='init_max')

    init_min_max = tf.group(init_min, init_max, name='init_min_max')

    return init_min_max


def minmax_quantization(num_bits, zero_passthrough, sign_passthrough):
    """ Simple quantization configuration for linear interpolation between
    trained minimum and maximum values.

    Parameters
    ----------
    num_bits: The number of bits to use for quantization.
    zero_passthrough: Whether to preserve exact zero values.
    sign_passthrough: Whether to encode positive and negative values separately.

    Returns
    -------
    config: A dictionary giving the configuration for minimum and maximum values.
    """
    def _config(variable):
        return make_minmax_config(variable, num_bits, zero_passthrough, sign_passthrough)

    return {
        'config': _config,
        'data': make_minmax_data,
        'quantization': make_minmax_quantization,
        'init': make_minmax_init
    }
