# Modifications Copyright 2018 Wenda Zhou. All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to preprocess images for the Inception networks."""

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.

    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image_encoded,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.5, 1.0),
                                max_attempts=100,
                                scope=None,
                                fuse_decode_and_crop=True):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Parameters
    ----------
    image_encoded: A string tensor containing the image encoded in jpeg format.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional scope for name_scope.
    fuse_decode_and_crop: Whether to use the fused decode and crop operation.

    Returns
    -------
    image: A 3-d tensor representing the cropped image.
    distort_bbox: the distorted bounding box.
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_encoded, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_encoded),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        if fuse_decode_and_crop:
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
            image = tf.image.decode_and_crop_jpeg(
                image_encoded, crop_window,
                channels=3, dct_method='INTEGER_FAST')
        else:
            image = tf.image.decode_jpeg(image_encoded, channels=3, dct_method='INTEGER_FAST')
            image = tf.slice(image, bbox_begin, bbox_size)

        return image, distort_bbox


def preprocess_for_train(record, height, width,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Parameters
    ----------
    record: A dictionary containing attributes of the example.
    height: height of the processed image in pixels
    width: width of the processed image in pixels
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
        bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.

    Returns
    -------
    A rank 3 tensor representing the processed image.
    """
    image_encoded = record['image/encoded']

    bbox = record['image/object/bbox']

    with tf.name_scope(scope, 'distort_image', [height, width, bbox]):
        distorted_image, distorted_bbox = distorted_bounding_box_crop(image_encoded, bbox)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([None, None, 3])

        # This resizing operation may distort the images because the aspect
        # ratio is not respected.
        distorted_image = tf.image.resize_images(
            distorted_image, [height, width], 1, align_corners=False)

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Restore the shape as dynamic slice may change shape.
        distorted_image.set_shape([height, width, 3])

        if add_image_summaries:
            tf.summary.image('cropped_resized_maybe_flipped_image',
                             tf.expand_dims(distorted_image, 0))

        # Now convert image to floating point
        distorted_image = tf.cast(distorted_image, dtype=tf.float32)
        distorted_image /= 255.

        # Randomly distort the colors. There are 4 ways to do it.
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering, fast_mode),
            num_cases=4)

        if add_image_summaries:
            tf.summary.image('final_distorted_image', tf.expand_dims(distorted_image, 0))

        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)
        return distorted_image


def _decode_and_central_crop(image_encoded, central_fraction):
    image_shape = tf.image.extract_jpeg_shape(image_encoded)

    img_h = tf.cast(image_shape[0], tf.float32)
    img_w = tf.cast(image_shape[1], tf.float32)
    bbox_h_start = tf.cast((img_h - img_h * central_fraction) / 2, tf.int32)
    bbox_w_start = tf.cast((img_w - img_w * central_fraction) / 2, tf.int32)

    bbox_h_size = image_shape[0] - bbox_h_start * 2
    bbox_w_size = image_shape[1] - bbox_w_start * 2

    crop_window = tf.stack([bbox_h_start, bbox_w_start, bbox_h_size, bbox_w_size])

    image = tf.image.decode_and_crop_jpeg(
        image_encoded, crop_window, channels=3, dct_method='INTEGER_FAST')

    return image


def preprocess_for_eval(record, height: int, width: int,
                        central_fraction=0.875, scope=None):
    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Parameters
    ----------
    record: A dictionary containing attributes of the example.
    height: height of the processed image in pixels.
    width: width of the processed image in pixels.
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.

    Returns
    -------
    A rank 3 tensor representing the processed image.
    """
    image_encoded = record['image/encoded']

    with tf.name_scope(scope, 'eval_image'):
        if central_fraction:
            image = _decode_and_central_crop(image_encoded, central_fraction=central_fraction)
        else:
            image = tf.image.decode_jpeg(image_encoded, channels=3, dct_method='INTEGER_FAST')

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])

        image = tf.cast(image, tf.float32) / 255
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def preprocess_image(record, height, width,
                     is_training=False,
                     fast_mode=True,
                     add_image_summaries=False):
    """Pre-process one image for training or evaluation.

    Parameters
    ----------
    record: record object containing all information on the example.
    height: height of the processed image in pixels.
    width: width of the processed image in pixels.
    is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
    fast_mode: Optional boolean, if True avoids slower transformations.
    add_image_summaries: Enable image summaries.

    Returns
    -------
    3-D float Tensor containing an appropriately scaled image
    """
    if is_training:
        return preprocess_for_train(record, height, width, fast_mode,
                                    add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(record, height, width)
