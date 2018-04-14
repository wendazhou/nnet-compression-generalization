""" Provides data for the Imagenet (ILSVRC 2012) dataset. """

import tensorflow as tf

from . import DatasetFactory

SPLITS_TO_SIZES = {'train': 1281167, 'test': 50000}

_NUM_CLASSES = 1001

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}


def _parse_record(record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(
            dtype=tf.int64),
    }

    features = tf.parse_single_example(record, keys_to_features)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return {
        'image/encoded': features['image/encoded'],
        'image/class/label': features['image/class/label'],
        'image/class/text': features['image/class/text'],
        'image/object/bbox': bbox
    }


def get_split(file_pattern, n_shards=4, shuffle=False) -> DatasetFactory:
    """Gets a dataset tuple with instructions for reading cifar10.

    The dataset contains tuples of images of various sizes and labels
    (integer value from 1 to 1000).

    Parameters
    ----------
    file_pattern: pattern of tfrecord files to match.
    n_shards: the number of files to read in parallel.
    shuffle: If true, shuffles the order in which the files are read.

    Returns
    -------
    dataset_factory: A function which creates a `tensorflow.data.Dataset` containing the data.
    """
    def imagenet_dataset_factory():
        dataset_fs = tf.data.Dataset.list_files(file_pattern)

        if shuffle:
            dataset_fs = dataset_fs.shuffle(1024, reshuffle_each_iteration=True)

        dataset = dataset_fs.apply(
            tf.contrib.data.parallel_interleave(
                lambda f: tf.data.TFRecordDataset(f, buffer_size=8 * 1024 * 1024),
                cycle_length=n_shards,
                sloppy=True,
            )
        )

        return dataset.map(_parse_record, num_parallel_calls=n_shards)

    return imagenet_dataset_factory
