""" Module for custom layers code.

"""

import tensorflow as tf
import contextlib


def _get_mixed_precision(x, weights, mixed_precision):
    if mixed_precision:
        weights = tf.cast(weights, tf.float16, name='cast_weights_fp16')
        _ensure_mixed_precision(x, mixed_precision, warn=True)

    return x, weights


def _ensure_mixed_precision(x, mixed_precision, warn=False):
    if not mixed_precision:
        return x

    if x.dtype != tf.float16:
        x = tf.cast(x, tf.float16)

        if warn:
            tf.logging.warn('Explicit conversion to fp16 inserted!')

    return x


def _get_scope(name, default_name):
    if name != '':
        return tf.variable_scope(name, default_name)
    else:
        return contextlib.suppress()


def conv2d(x, filters, kernel_size,
           stride=1, padding='SAME',
           mixed_precision=False,
           data_format='channels_first',
           name=None,
           variable_name='weights'):
    """ Convolutional layer.

    Parameters
    ----------
    x: The input of the layer.
    filters: The number of output filters.
    kernel_size: The size of the kernel.
    stride: The stride of the kernel.
    padding: The padding mode of the convolution.
    mixed_precision: Whether to run a mixed-precision convolution.
    data_format: The data format of the input, either channels_first or channels_last.
    name: The name of the operation.
    variable_name: The name of the weights variable.

    Returns
    -------
    The output of the convolution.
    """

    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError('data_format must be "channels_first" or "channels_last".')

    if isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    if data_format == 'channels_first':
        channels_in = x.shape[1]
        strides = [1, 1] + stride
        df = 'NCHW'
    else:
        channels_in = x.shape[-1]
        strides = [1] + stride + [1]
        df = 'NHWC'

    with _get_scope(name, 'conv2d'):
        weights = tf.get_variable(variable_name,
                                  shape=list(kernel_size) + [channels_in, filters],
                                  dtype=tf.float32,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.WEIGHTS,
                                               tf.GraphKeys.MODEL_VARIABLES],
                                  trainable=True)

        x, weights = _get_mixed_precision(x, weights, mixed_precision)

        x = tf.nn.conv2d(
            x, weights,
            strides=strides, padding=padding,
            data_format=df, name='convolution')

    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return x


def depthwise_conv2d(x, depth_multiplier, kernel_size, stride=1, padding='SAME',
                     mixed_precision=False, data_format='channels_first',
                     name=None):

    if data_format not in ['channels_first', 'channels_last']:
        raise ValueError('data_format must be "channels_first" or "channels_last".')

    if isinstance(stride, int):
        stride = [stride, stride]

    if data_format == 'channels_first':
        channels_in = x.shape[1]
        strides = [1, 1] + stride
        df = 'NCHW'
    else:
        channels_in = x.shape[-1]
        strides = [1] + stride + [1]
        df = 'NHWC'

    with _get_scope(name, 'conv2d'):
        weights = tf.get_variable('weights',
                                  shape=list(kernel_size) + [channels_in, depth_multiplier],
                                  dtype=tf.float32,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.WEIGHTS,
                                               tf.GraphKeys.MODEL_VARIABLES],
                                  trainable=True)

        x, weights = _get_mixed_precision(x, weights, mixed_precision)

        x = tf.nn.depthwise_conv2d(
            x, weights, strides, padding,
            data_format=df, name='convolution'
        )

    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, x)
    return x


def dense(x, units, use_bias=True, mixed_precision=False,
          name=None,
          weights_name='kernel',
          bias_name='bias'):
    """ Dense layer

    Parameters
    ----------
    x: The input to the layer.
    units: The number of output units.
    use_bias: Whether to add a bias.
    mixed_precision: Whether to use mixed precision training.
    name: The name of the layer.
    weights_name: The name of the weights variable
    bias_name: The name of the bias variable

    Returns
    -------
    x: The output of the layer.
    """
    channels_in = x.shape[-1]

    with _get_scope(name, 'dense'):
        weights = tf.get_variable(weights_name,
                                  shape=[channels_in, units],
                                  dtype=tf.float32,
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                               tf.GraphKeys.WEIGHTS,
                                               tf.GraphKeys.MODEL_VARIABLES],
                                  trainable=True)
        weights = _ensure_mixed_precision(weights, mixed_precision, warn=False)
        x = _ensure_mixed_precision(x, mixed_precision, warn=True)

        x = tf.matmul(x, weights)

        if use_bias:
            bias = tf.get_variable(bias_name,
                                   shape=[units],
                                   dtype=tf.float32,
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                tf.GraphKeys.BIASES,
                                                tf.GraphKeys.MODEL_VARIABLES],
                                   trainable=True,
                                   initializer=tf.zeros_initializer())

            bias = _ensure_mixed_precision(bias, mixed_precision, warn=False)
            x = tf.nn.bias_add(x, bias)

    return x


def batch_normalization(x, is_training, momentum=0.9997, epsilon=0.001,
                        data_format='channels_first'):
    """ Batch normalization layer.

    Parameters
    ----------
    x: The input to the layer.
    is_training: If true, create operations to update the moving mean and variance.
    momentum: The momentum parameter.
    epsilon: The epsilon decay parameter.
    data_format: The data format of the input.

    Returns
    -------
    The result of batch normalization.
    """
    if data_format == 'channels_first':
        axis = 1
    elif data_format == 'channels_last':
        axis = -1
    else:
        raise ValueError('Invalid data_format')

    batch_norm = tf.layers.BatchNormalization(
        axis=axis, momentum=momentum, epsilon=epsilon,
        fused=True)

    batch_norm.build(x.shape)

    for v in batch_norm.variables:
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)

    x = batch_norm(x, training=is_training)

    return x


def format_input(x, mixed_precision, data_format):
    """ Converts the input into the given precision and data format.

    Parameters
    ----------
    x: The input tensor.
    mixed_precision: If True, converts the input to a half precision floating point.
    data_format: If 'channels_first', permute the data to NCHW format.

    Returns
    -------
    x: The corresponding tensor.
    """
    if mixed_precision:
        x = tf.cast(x, tf.float16, name='cast_fp16')

    if data_format == 'channels_first':
        x = tf.transpose(x, [0, 3, 1, 2], name='transpose_nchw')

    return x
