"""Provides data for the Cifar10 dataset, but randomises the labels

Loads the data using the new DataSet API. This completely
randomises the label by creating a label which corresponds
to the hash value of the image.
"""

import numpy as np
import tensorflow as tf
import functools
from . import DatasetFactory


_FILE_PATTERN = 'cifar10_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}


def _get_random_labels_and_probability(num_labels, seed):
    random = np.random.RandomState(seed)

    random_labels = random.randint(num_labels, size=50000)
    random_probabilities = random.random_sample(size=50000)

    data = tf.data.Dataset.from_tensor_slices(
        {'random_label': random_labels, 'random_prob': random_probabilities})

    return data


def _parse_record(record, random_data, random_prob=1):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    label = tf.where(
        tf.less_equal(random_data['random_prob'], random_prob),
        tf.cast(random_data['random_label'], dtype=tf.int64),
        parsed['image/class/label'],
        name='label')

    image = tf.image.decode_png(parsed['image/encoded'], channels=3)

    return {
        'image': image,
        'image/class/label': label,
        'image/class/original_label': parsed['image/class/label']
    }


def get_split(record_file, num_classes: int=10,
              proportion_random: float=1,
              seed=42) -> DatasetFactory:
    """Gets a dataset of CIFAR-10 data with randomized labels.

    This gets a dataset of images from the given record, with the labels
    corresponding to the images being completely randomized.

    Parameters
    ----------
    record_file: The path to the record file containing the images.
    num_classes: The number of classes to split the data into.
    proportion_random: The proportion of instances that are randomised.
    seed: The random seed to use.

    Returns
    -------
    dataset: A `tensorflow.data.Dataset` containing the images with
        randomized labels.
    """
    def cifar_dataset_factory():
        parse_fn = functools.partial(_parse_record,
                                     random_prob=proportion_random)

        dataset = tf.data.TFRecordDataset(record_file)
        random_data = _get_random_labels_and_probability(num_classes, seed)

        return (tf.data.Dataset.zip((dataset, random_data))
                .map(parse_fn, num_parallel_calls=4)
                .cache())

    return cifar_dataset_factory
