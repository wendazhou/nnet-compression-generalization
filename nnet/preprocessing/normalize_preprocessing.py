"""
This module defines a minimal preprocessing routine that only
normalizes the image and does no additional preprocessing.
"""

import tensorflow as tf


def preprocess_for_train(record, output_height, output_width, add_image_summaries=False, center=False):
    """Normalization only preprocessing. """

    if 'image' in record:
        image = record['image']
    elif 'image/encoded' in record:
        image = tf.image.decode_jpeg(record['image/encoded'], channels=3, dct_method='INTEGER_FAST')
    else:
        raise ValueError('Neither "image" nor "image/encoded" keys found in record.')

    if add_image_summaries:
        tf.summary.image('image', tf.expand_dims(image, 0))

    if image.dtype not in [tf.float32, tf.float64]:
        image = tf.divide(tf.to_float(image), 255)

    image = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)

    if center:
    # We scale the image from -1 to 1 instead of 0 to 1.
        image = 2 * (image - 0.5)

    return image


def preprocess_for_eval(record, output_height, output_width, center=False):
    return preprocess_for_train(record, output_height, output_width, center)


def preprocess(image, output_height, output_width, training=True, add_image_summaries=False, center=False):
    if training:
        return preprocess_for_train(image, output_height, output_width, add_image_summaries, center)
    else:
        return preprocess_for_eval(image, output_height, output_width, center)


__all__ = ['preprocess_for_train', 'preprocess_for_eval', 'preprocess']
