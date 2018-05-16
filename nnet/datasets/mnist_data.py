""" Provides data for the MNIST dataset. """

import tensorflow as tf
import os

from . import DatasetFactory


def _decode_image(image):
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    return image


def _decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [])
    return tf.to_int32(label)


def _make_record(image, label):
    return {
        'image': image,
        'image/class/label': label
    }


def _maybe_decompress(file):
    if tf.gfile.Exists(file):
        return file

    import shutil
    import gzip

    if not tf.gfile.Exists(file + '.gz'):
        raise ValueError('File {0} not found and no compressed version'.format(file))

    with gzip.open(file + '.gz', 'rb') as f_in, open(file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    return file


def _load_dataset(directory, image_name, label_name) -> DatasetFactory:
    def mnist_dataset_factory() -> tf.data.Dataset:
        images = tf.data.FixedLengthRecordDataset(
            _maybe_decompress(os.path.join(directory, image_name)),
            28 * 28, header_bytes=16
        ).map(_decode_image)

        labels = tf.data.FixedLengthRecordDataset(
            _maybe_decompress(os.path.join(directory, label_name)),
            1, header_bytes=8
        ).map(_decode_label)

        return tf.data.Dataset.zip((images, labels)).map(_make_record).cache()

    return mnist_dataset_factory


def get_split(directory, train_or_test='train') -> DatasetFactory:
    """ Gets a split (train or test) of the MNIST dataset.

    Parameters
    ----------
    directory: The directory in which the MNIST dataset is contained.
    train_or_test: A string "train" or "test" indicating whether the training
        or testing dataset is required.

    Returns
    -------
    A `tf.data.Dateset` of tuples containing the image and the label.
    """
    if train_or_test == 'train':
        return _load_dataset(directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    elif train_or_test == 'test':
        return _load_dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    else:
        raise ValueError('train_or_test must be either "train" or "test"')
