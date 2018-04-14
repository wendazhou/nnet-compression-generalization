# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright (c) 2017 Wenda Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import contextlib
from . import layers

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    inputs = layers.batch_normalization(
        inputs, is_training,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        data_format=data_format)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Parameters
    ----------
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
    data_format: The data format of the input.

    Returns
    -------
    A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    if kernel_size == 1:
        return inputs

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        pad = [[0, 0]] * 2 + [[pad_beg, pad_end]] * 2
    elif data_format == 'channels_last':
        pad = [[0, 0]] + [[pad_beg, pad_end]] * 2 + [[0, 0]]
    else:
        raise ValueError('Invalid input format.')

    padded_inputs = tf.pad(inputs, pad)
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         mixed_precision=False, data_format='channels_first'):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    padding = 'SAME' if strides == 1 else 'VALID'

    return layers.conv2d(inputs, filters, kernel_size, strides, padding,
                         mixed_precision, data_format,
                         variable_name='kernel_weights')


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   mixed_precision=False, data_format='channels_first'):
    """Standard building block for residual networks with BN before convolutions.

    Parameters
    ----------
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
        a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
    mixed_precision: Whether to use mixed-precision training.
    data_format: The data format to use.

    Returns
    -------
    The output tensor of the block.
    """

    with tf.variable_scope('residual_block'):
        shortcut = inputs

        with tf.variable_scope('conv_1'):
            inputs = batch_norm_relu(inputs, is_training, data_format)

            # The projection shortcut should come after the first batch norm and ReLU
            # since it performs a 1x1 convolution.
            if projection_shortcut is not None:
                with tf.variable_scope('shortcut'):
                    shortcut = projection_shortcut(inputs)

            inputs = conv2d_fixed_padding(
                inputs, filters,
                kernel_size=3, strides=strides,
                mixed_precision=mixed_precision, data_format=data_format)

        with tf.variable_scope('conv_2'):
            inputs = batch_norm_relu(inputs, is_training, data_format)

            inputs = conv2d_fixed_padding(
                inputs, filters,
                kernel_size=3, strides=1,
                mixed_precision=mixed_precision, data_format=data_format)

    return tf.add(inputs, shortcut, name='output')


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides,
                     mixed_precision=False, data_format='channels_first'):
    """Bottleneck block variant for residual networks with BN before convolutions.

    Parameters
    ----------
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
        third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
        a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
    mixed_precision: Whether to use mixed precision training.
    data_format: The data format of the input.

    Returns
    -------
    The output tensor of the block.
    """

    with tf.variable_scope('bottleneck_block'):
        shortcut = inputs
        inputs = batch_norm_relu(inputs, is_training, data_format)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = conv2d_fixed_padding(
            inputs, filters,
            kernel_size=1, strides=1,
            mixed_precision=mixed_precision, data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = conv2d_fixed_padding(
            inputs, filters,
            kernel_size=3, strides=strides,
            mixed_precision=mixed_precision, data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = conv2d_fixed_padding(
            inputs, 4 * filters,
            kernel_size=1, strides=1,
            mixed_precision=mixed_precision, data_format=data_format)

    return tf.add(inputs, shortcut, name='output')


def block_layer(inputs, filters, block_fn, blocks, strides, is_training,
                mixed_precision=False, data_format='channels_first',
                name=None):
    """Creates one layer of blocks for the ResNet model.

    Parameters
    ----------
    inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
    mixed_precision: Whether to use mixed-precision training.
    data_format: The data format of the inputs.
    name: A string name for the tensor output of the block layer.

    Returns
    -------
    The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(x):
        return conv2d_fixed_padding(
            inputs=x, filters=filters_out,
            kernel_size=1, strides=strides,
            mixed_precision=mixed_precision, data_format=data_format)

    with tf.variable_scope(name, 'block_layer'):
        with tf.variable_scope('block_init'):
            # Only the first block per block_layer uses projection_shortcut and strides
            inputs = block_fn(
                inputs, filters, is_training, projection_shortcut, strides,
                mixed_precision=mixed_precision, data_format=data_format)

        for i in range(1, blocks):
            with tf.variable_scope('block_{0}'.format(i)):
                inputs = block_fn(inputs, filters, is_training, None, 1,
                                  mixed_precision=mixed_precision,
                                  data_format=data_format)

        return tf.identity(inputs, 'output')


def make_final_block(inputs, is_training, num_classes, mixed_precision, data_format):
    with tf.variable_scope('final'):
        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=8, strides=1, padding='VALID',
            data_format=data_format)

        inputs = tf.identity(inputs, 'final_avg_pool')

        n_final = np.prod(inputs.shape.as_list()[1:])

        inputs = tf.reshape(inputs, [-1, n_final])
        inputs = layers.dense(inputs, num_classes, mixed_precision=mixed_precision)

    return inputs


def cifar10_resnet_v2_generator(resnet_size: int,
                                num_classes: int,
                                model_scope=None):
    """Generator for CIFAR-10 ResNet v2 models.

    This generates a model with 3 layer blocks, with filter sizes
    in each block being respectively:
    - 32 x 32 x 16
    - 16 x 16 x 32
    - 8 x 8 x 64

    The layers use the simple residual block with preactivation, and each
    block layers contains (resnet_size - 2) / 6 blocks.

    The naming is as follows:
    block_layer1/block_1/residual_block/conv_1/conv2d/convolution:0

    Parameters
    ----------
    resnet_size: A single integer for the size of the ResNet model.
    num_classes: The number of possible classes for image classification.
    model_scope: A single global scope for the model.

    Returns
    -------
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.

    Raises
    ------
    ValueError: If `resnet_size` is invalid.
    """
    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    def model(inputs, is_training, mixed_precision=False, data_format='channels_first'):
        """Constructs the ResNet model given the inputs.

        Parameters
        ----------
        inputs: the input to the network.
        is_training: whether the network should be training.
        mixed_precision: whether to use mixed precision convolution.
        data_format: the data format to use for convolutions.

        Returns
        -------
        logits: the logits output by the network.
        """
        if model_scope is not None:
            scope = model_scope()
        else:
            scope = contextlib.suppress()

        with scope:
            inputs = layers.format_input(inputs, mixed_precision, data_format)

            with tf.variable_scope('initial_conv'):
                inputs = conv2d_fixed_padding(
                    inputs=inputs, filters=16, kernel_size=3, strides=1,
                    mixed_precision=mixed_precision, data_format=data_format)

                inputs = tf.identity(inputs, name='output')

            for i in range(3):
                strides = 1 if i == 0 else 2
                filters = int(16 * (2 ** i))
                inputs = block_layer(
                    inputs=inputs, filters=filters, block_fn=building_block, blocks=num_blocks,
                    strides=strides, is_training=is_training,
                    mixed_precision=mixed_precision, data_format=data_format,
                    name='block_layer{0}'.format(i + 1))

            inputs = make_final_block(inputs, is_training, num_classes,
                                      mixed_precision, data_format)

            inputs = tf.cast(inputs, tf.float32)

        return inputs

    model.default_image_size = 32
    return model


def imagenet_resnet_v2_generator(block_fn, block_size, num_classes):
    """Generator for ImageNet ResNet v2 models.

    Parameters
    ----------
    block_fn: The block to use within the model, either `building_block` or `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
        layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.

    Returns
    -------
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
    """

    def model(inputs, is_training, mixed_precision=False, data_format='channel_first'):
        """Constructs the ResNet model given the inputs.

        Parameters
        ----------
        inputs: the inputs to the network.
        is_training: whether the network is in training mode.
        mixed_precision: whether to use mixed precision for convolutions
        data_format: which data format to use for convolutions.
        """
        inputs = layers.format_input(inputs, mixed_precision, data_format)

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2,
            mixed_precision=mixed_precision, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        for i, (strides, filters) in enumerate([(1, 64), (2, 128), (2, 256), (2, 512)]):
            inputs = block_layer(
                inputs=inputs, filters=filters, block_fn=block_fn, blocks=block_size[i],
                strides=strides, is_training=is_training,
                mixed_precision=mixed_precision, data_format=data_format,
                name='block_layer{0}'.format(i + 1))

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=7, strides=1, padding='VALID', data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
        inputs = layers.dense(inputs, units=num_classes, mixed_precision=mixed_precision)
        inputs = tf.identity(inputs, 'final_dense')
        return tf.cast(inputs, dtype=tf.float32)

    model.default_image_size = 224
    return model


def resnet_v2(resnet_size, num_classes):
    """Returns the ResNet model for a given size and number of output classes, as designed
    in the original paper: https://arxiv.org/pdf/1512.03385.pdf

    The supported sizes are 18, 34, 50, 101, 152 and 200. Unlike the original paper,
    these models uses the full preactivation model (as in the follow up paper), but
    otherwise follow the original design.

    Parameters
    ----------
    resnet_size: The number of layers in the network. Must be one of the pre-configured values.
    num_classes: The number of classes.

    Returns
    -------
    model_fn: A function that when called with inputs, generates the logits.
    """
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    params = model_params[resnet_size]
    return imagenet_resnet_v2_generator(params['block'], params['layers'], num_classes)
