""" Utilities to load the common image classification datasets.

This package provides functionality to load the MNIST, CIFAR-10 and ImageNet datasets.

"""

import typing
import tensorflow as tf

DatasetFactory = typing.Callable[[], tf.data.Dataset]