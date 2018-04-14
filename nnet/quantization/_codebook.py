""" Codebook based quantization.

This module implements quantization that is based around a codebook instead of linearly
spread across a range of values. This is more expensive to train, but can permit
a even lower amount of bits.

"""

import tensorflow as tf
import numpy as np

from functools import partial


def _evaluate_fn(fn_or_value, variable):
    if callable(fn_or_value):
        return fn_or_value(variable)
    else:
        return fn_or_value


def make_codebook_config(variable, num_bits, zero_passthrough):
    num_bits = _evaluate_fn(num_bits, variable)
    zero_passthrough = _evaluate_fn(zero_passthrough, variable)

    num_clusters = 2 ** num_bits - int(zero_passthrough)

    return {
        'variable': variable,
        'num_bits': num_bits,
        'zero_passthrough': zero_passthrough,
        'shape': variable.shape,
        'num_clusters': num_clusters
    }


def make_codebook_data(config, data_collections):
    variable = config['variable']
    shape = config['shape']
    num_bits = config['num_bits']
    zero_passthrough = config['zero_passthrough']

    num_clusters = 2 ** num_bits

    if zero_passthrough:
        num_clusters = num_clusters - 1

    labels = tf.get_variable('cluster_labels', shape=shape, dtype=tf.int32,
                             initializer=tf.zeros_initializer(),
                             collections=data_collections,
                             trainable=False)

    codebook = tf.get_variable('cluster_values', shape=[num_clusters],
                               dtype=variable.dtype,
                               initializer=tf.zeros_initializer(),
                               collections=data_collections,
                               trainable=True)

    if zero_passthrough:
        mask = tf.get_variable('mask', shape=shape, dtype=variable.dtype,
                               trainable=False,
                               collections=data_collections,
                               initializer=tf.zeros_initializer())
    else:
        mask = None

    return {
        'labels': labels,
        'codebook': codebook,
        'mask': mask,
    }


def make_codebook_quantization(config, data):
    labels = data['labels']
    codebook = data['codebook']
    mask = data['mask']

    quantized_variable = tf.gather(codebook, labels)

    if mask is not None:
        quantized_variable = tf.multiply(quantized_variable, mask)

    return quantized_variable


def make_codebook_init(config, data):
    from sklearn.cluster import KMeans

    labels = data['labels']
    cluster_centers = data['codebook']
    num_clusters = config['num_clusters']

    def fn(nonzero_values):
        if np.size(nonzero_values) < num_clusters:
            clusters, labs = np.unique(nonzero_values, return_inverse=True)
        else:
            kmeans = KMeans(num_clusters)
            kmeans.fit(nonzero_values.reshape(-1, 1))
            labs = kmeans.labels_
            clusters = kmeans.cluster_centers_.reshape(-1)

        return labs.astype(np.int32, copy=True), clusters.astype(np.float32, copy=True)

    variable = config['variable']
    zero_passthrough = config['zero_passthrough']
    variable_flat = tf.reshape(variable, [-1])

    if zero_passthrough:
        idx_nonzero = tf.where(tf.not_equal(variable_flat, 0.0))
        variable_flat = tf.gather(variable_flat, tf.reshape(idx_nonzero, [-1]))

    labels_init, clusters_init = tf.py_func(fn, [variable_flat], [tf.int32, tf.float32], stateful=False)

    if zero_passthrough:
        labels_idx = tf.where(tf.not_equal(variable, 0.0))
        assign_labels = tf.scatter_nd_update(
            labels, labels_idx, labels_init,
            name='assign_nonzero_labels')
    else:
        assign_labels = tf.assign(labels, tf.reshape(labels_init, labels.shape))

    assign_clusters = tf.scatter_update(
        cluster_centers, tf.range(tf.size(clusters_init)), clusters_init,
        name='assign_nonzero_clusters')

    if zero_passthrough:
        mask = data['mask']
        assign_mask = tf.assign(mask, tf.to_float(tf.not_equal(variable, 0.0)),
                                name='assign_mask')
    else:
        assign_mask = tf.no_op()

    return tf.group(assign_labels, assign_clusters, assign_mask, name='init_codebook')


def codebook_quantization(num_bits, zero_passthrough):
    """ Quantization configuration for codebook based quantization.

    Parameters
    ----------
    num_bits: The number of bits to encode.
    zero_passthrough: Whether to encode an exact zero.

    Returns
    -------
    config: A dictionary giving the configuration for codebook based quantization.
    """
    return {
        'config': partial(make_codebook_config, num_bits=num_bits, zero_passthrough=zero_passthrough),
        'data': make_codebook_data,
        'quantization': make_codebook_quantization,
        'init': make_codebook_init
    }
