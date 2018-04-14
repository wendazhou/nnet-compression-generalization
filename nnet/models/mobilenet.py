# Modifications Copyright 2018 Wenda Zhou. All Rights Reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileNet v1.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1704.04861.

  MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam

100% Mobilenet V1 (base) with input size 224x224:

See mobilenet_v1()

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 864      10,838,016
MobilenetV1/Conv2d_1_depthwise/depthwise:                    288       3,612,672
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     2,048      25,690,112
MobilenetV1/Conv2d_2_depthwise/depthwise:                    576       1,806,336
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     8,192      25,690,112
MobilenetV1/Conv2d_3_depthwise/depthwise:                  1,152       3,612,672
MobilenetV1/Conv2d_3_pointwise/Conv2D:                    16,384      51,380,224
MobilenetV1/Conv2d_4_depthwise/depthwise:                  1,152         903,168
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    32,768      25,690,112
MobilenetV1/Conv2d_5_depthwise/depthwise:                  2,304       1,806,336
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    65,536      51,380,224
MobilenetV1/Conv2d_6_depthwise/depthwise:                  2,304         451,584
MobilenetV1/Conv2d_6_pointwise/Conv2D:                   131,072      25,690,112
MobilenetV1/Conv2d_7_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_8_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_9_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_10_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_11_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_12_depthwise/depthwise:                 4,608         225,792
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  524,288      25,690,112
MobilenetV1/Conv2d_13_depthwise/depthwise:                 9,216         451,584
MobilenetV1/Conv2d_13_pointwise/Conv2D:                1,048,576      51,380,224
--------------------------------------------------------------------------------
Total:                                                 3,185,088     567,716,352


75% Mobilenet V1 (base) with input size 128x128:

See mobilenet_v1_075()

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 648       2,654,208
MobilenetV1/Conv2d_1_depthwise/depthwise:                    216         884,736
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     1,152       4,718,592
MobilenetV1/Conv2d_2_depthwise/depthwise:                    432         442,368
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     4,608       4,718,592
MobilenetV1/Conv2d_3_depthwise/depthwise:                    864         884,736
MobilenetV1/Conv2d_3_pointwise/Conv2D:                     9,216       9,437,184
MobilenetV1/Conv2d_4_depthwise/depthwise:                    864         221,184
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    18,432       4,718,592
MobilenetV1/Conv2d_5_depthwise/depthwise:                  1,728         442,368
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    36,864       9,437,184
MobilenetV1/Conv2d_6_depthwise/depthwise:                  1,728         110,592
MobilenetV1/Conv2d_6_pointwise/Conv2D:                    73,728       4,718,592
MobilenetV1/Conv2d_7_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_8_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_9_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_10_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_11_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_12_depthwise/depthwise:                 3,456          55,296
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  294,912       4,718,592
MobilenetV1/Conv2d_13_depthwise/depthwise:                 6,912         110,592
MobilenetV1/Conv2d_13_pointwise/Conv2D:                  589,824       9,437,184
--------------------------------------------------------------------------------
Total:                                                 1,800,144     106,002,432

"""

import functools
from collections import namedtuple

import tensorflow as tf

from . import layers

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]


def _batch_norm_and_relu(x, is_training, data_format='channels_first'):
    _default_momentum = 0.996
    _retrain_momentum = 0.95

    x = layers.batch_normalization(x, is_training,
                                   momentum=_default_momentum, epsilon=0.001,
                                   data_format=data_format)

    x = tf.nn.relu6(x)
    return x


def convolutional_layer(x, depth, kernel_size, strides,
                        is_training=True,
                        mixed_precision=False,
                        data_format='channels_first',
                        name=None):
    """ Standard convolutional layer.

    Parameters
    ----------
    x: network input
    depth: number of layers
    kernel_size: convolutional kernel size
    strides: strides of the convolution
    is_training: whether the layer is instantiated for training or testing.
    mixed_precision: whether to use mixed precision convolutions.
    data_format: the data format of the input.
    name: name of the layer.

    Returns
    -------
    x: the output of applying this convolutional layer.
    """
    with tf.variable_scope(name, default_name='Conv2d'):
        x = layers.conv2d(x, depth, kernel_size, strides,
                          mixed_precision=mixed_precision,
                          data_format=data_format,
                          name='')
        x = _batch_norm_and_relu(x, is_training, data_format)

    return x


def separable_layer(x, depth, kernel_size, stride,
                    is_training=True, mixed_precision=False,
                    data_format='channels_first',
                    name=None):
    """ Separable convolutional layer.

    A layer that is composed of a depthwise layer + batch norm + non-linearity,
    and then a pointwise layer + batch norm + non-linearity.

    Parameters
    ----------
    x: network input
    depth: number of layers
    kernel_size: convolutional kernel size
    stride: stride of the convolution
    is_training: whether the layer is instantiated for training or testing.
    mixed_precision: whether to use mixed precision convolutions.
    data_format: the data format of the inputs.
    name: name of the layer

    Returns
    -------
    x: the output of applying this separable convolutional layer.
    """
    with tf.variable_scope(name, default_name='SeparableConv2d'):
        with tf.variable_scope('depthwise'):
            x = layers.depthwise_conv2d(x, 1, kernel_size, stride,
                                        mixed_precision=mixed_precision,
                                        data_format=data_format,
                                        name='')
            x = _batch_norm_and_relu(x, is_training, data_format)

        with tf.variable_scope('pointwise'):
            x = layers.conv2d(x, depth, [1, 1], 1,
                              mixed_precision=mixed_precision,
                              data_format=data_format,
                              name='')
            x = _batch_norm_and_relu(x, is_training, data_format)

    return x


def final_layer(x, num_classes, data_format='channels_first'):
    if data_format == 'channels_first':
        df = 'NCHW'
        spatial_axes = [2, 3]
    elif data_format == 'channels_last':
        df = 'NHWC'
        spatial_axes = [1, 2]
    else:
        raise ValueError('Invalid data format.')

    with tf.variable_scope('conv2d_final'):
        x = layers.conv2d(x, num_classes, [1, 1], data_format=data_format,
                          name='', variable_name='kernel')

        biases = tf.get_variable(
            'bias',
            shape=[num_classes],
            dtype=tf.float32,
            collections=[
                tf.GraphKeys.GLOBAL_VARIABLES,
                tf.GraphKeys.MODEL_VARIABLES
            ],
            trainable=True)

        x = tf.nn.bias_add(x, biases, data_format=df)
        x = tf.squeeze(x, spatial_axes, name='SpatialSqueeze')

    return x


def mobilenet_v1_base(inputs,
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      is_training=True,
                      mixed_precision=False,
                      data_format='channels_first'):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs. This function expects
    its inputs to be provided in NCHW format.

    Parameters
    ----------
    inputs: a tensor of shape [batch_size, height, width, channels].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    is_training: Whether the network is constructed for training or evaluation.
    mixed_precision: Whether to use mixed-precision (fp16) convolutions.
    data_format: The data format of the convolutions.

    Returns
    -------
    tensor_out: output tensor corresponding to the final_endpoint.

    Raises
    ------
    ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0, or the target output_stride is not
                  allowed.
    """

    def compute_depth(d):
        return max(int(d * depth_multiplier), min_depth)

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    net = inputs

    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'Conv2d_%d' % i

        if isinstance(conv_def, Conv):
            net = convolutional_layer(
                net, compute_depth(conv_def.depth),
                conv_def.kernel, conv_def.stride,
                is_training, mixed_precision, data_format,
                name=end_point_base)

        elif isinstance(conv_def, DepthSepConv):
            net = separable_layer(
                net, compute_depth(conv_def.depth),
                conv_def.kernel, conv_def.stride,
                is_training, mixed_precision, data_format,
                name=end_point_base)
        else:
            raise ValueError('Unknown convolution type %s for layer %d'
                             % (conv_def.ltype, i))

    return net


def mobilenet_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 convolution_definitions=None,
                 mixed_precision=False,
                 data_format='channels_first',
                 scope='MobilenetV1'):
    """Mobilenet v1 model for classification.

    Parameters
    ----------
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer
        are returned instead.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
    convolution_definitions: A list of ConvDef namedtuples specifying the net architecture.
    mixed_precision: Whether to enable fp16 mixed precision training.
    data_format: The data format to use for the convolutions.
    scope: Optional variable_scope.

    Returns
    -------
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises
    ------
    ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    if data_format not in ('channels_first', 'channels_last'):
        raise ValueError('Invalid data format.')

    tf.logging.info('Creating MobilenetV1 (depth multiplier: %g, classes: %d, mixed precision=%s, data format: %s)',
                    depth_multiplier, num_classes, mixed_precision, data_format)

    if not callable(scope):
        scope_fn = lambda: tf.variable_scope(scope, 'MobilenetV1', [inputs])
    else:
        scope_fn = scope

    with scope_fn():
        net = inputs

        if mixed_precision:
            net = tf.cast(net, dtype=tf.float16, name='convert_fp16')

        if data_format == 'channels_first':
            net = tf.transpose(net, [0, 3, 1, 2], name='shuffle_nchw')

        net = mobilenet_v1_base(net,
                                min_depth=min_depth,
                                depth_multiplier=depth_multiplier,
                                conv_defs=convolution_definitions,
                                is_training=is_training,
                                mixed_precision=mixed_precision,
                                data_format=data_format)

        with tf.variable_scope('Logits'):
            if mixed_precision:
                net = tf.cast(net, dtype=tf.float32, name='convert_fp32')

            if data_format == 'channels_first':
                spatial_axes = [2, 3]
            else:
                spatial_axes = [1, 2]

            net = tf.reduce_mean(net, spatial_axes, keepdims=True, name='global_pool')

            if not num_classes:
                return net
            # 1024 x 1 x 1

            logits = final_layer(net, num_classes, data_format)
    return logits


mobilenet_v1.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


mobilenet_v1_075 = wrapped_partial(mobilenet_v1, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet_v1, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet_v1, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def rename_pretrained(name: str):
    """ Matches the name of a variable saved in the pre-trained MobileNet
    networks with the name of the corresponding variable in this network.

    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

    Parameters
    ----------
    name: the original name of the variable.

    Returns
    -------
    name: the new name of the variable.
    """
    import re

    replace_list = {
        r"BatchNorm": r"batch_normalization",
        r"Conv2d_(\d+)_(\w+)/": r"Conv2d_\1/\2/",
        r"Conv2d_1c_1x1/biases": r"conv2d_final/bias",
        r"Conv2d_1c_1x1/weights": r"conv2d_final/kernel",
        r"depthwise_weights": r"weights"
    }

    for match, replace in replace_list.items():
        name = re.sub(match, replace, name)

    return name
