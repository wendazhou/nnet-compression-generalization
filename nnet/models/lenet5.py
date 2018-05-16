""" Tensorflow implementation of LeNet-5 on MNIST

This is an implementation of the LeNet 5 network, which has two convolutional
and two fully connected layers.

"""

import tensorflow as tf
from . import layers


def _conv2d_layer(x, filters, kernel_size, data_format, name=None):
    if data_format == 'channels_first':
        df = 'NCHW'
    elif data_format == 'channels_last':
        df = 'NHWC'
    else:
        raise ValueError('Invalid data format')

    with tf.variable_scope(name, 'Conv2D'):
        biases = tf.get_variable(
            'bias',
            shape=[filters],
            dtype=tf.float32,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.BIASES, tf.GraphKeys.MODEL_VARIABLES],
            trainable=True
        )

        x = layers.conv2d(x, filters, kernel_size,
                          padding='VALID', data_format=data_format,
                          name='')

        x = tf.nn.bias_add(x, biases, data_format=df, name='bias')
    return x


def lenet(inputs,
          is_training=True,
          scope='LeNet',
          data_format=None):
    """ Implementation of the LeNet-5 network. It has two convolutional
    layers and two fully connected layers. As taken from the Caffe model zoo,
    it only has a single relu non-linearity between the two fully connected layers
    (and the max-pooling operation between convolutional layers).

    Parameters
    ----------
    inputs: The input image to the network.
    scope: An optional scope in which to create the network.
    data_format: The data-format of the network, either 'channel_first'
        or 'channel_last', or None to select automatically.

    Returns
    -------
    logits: The logits for the network.
    """
    if data_format is None:
        data_format = 'channels_first' if tf.test.is_gpu_available() else "channels_last"

    if data_format == 'channels_first':
        input_shape = [-1, 1, 28, 28]
    elif data_format == 'channels_last':
        input_shape = [-1, 28, 28, 1]
    else:
        raise ValueError('data_format must be either "channels_first", "channels_last" or None')

    x = tf.reshape(inputs, input_shape, name='reshape_inputs')

    if not callable(scope):
        scope_fn = lambda: tf.variable_scope(scope)
    else:
        scope_fn = scope

    with scope_fn():
        x = _conv2d_layer(x, 20, 5, data_format=data_format, name='conv1')
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name='pool1', data_format=data_format)
        x = _conv2d_layer(x, 50, 5, data_format=data_format, name='conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name='pool2', data_format=data_format)
        x = tf.layers.flatten(x)
        x = layers.dense(x, 500, name='fc1', weights_name='weights')
        x = tf.nn.relu(x, name='relu')
        x = layers.dense(x, 10, name='fc2', weights_name='weights')

        logits = tf.identity(x, name='logits')

        return logits
