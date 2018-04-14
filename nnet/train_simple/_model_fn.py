import inspect
from functools import partial

import tensorflow as tf


def warm_start(ckpt_to_start_from, ignore_missing=False):
    """ Warm start a given estimator by replacing the initializer.

    Parameters
    ----------
    ckpt_to_start_from: The checkpoint to restore from.
    ignore_missing: Whether to ignore variables not found in the given checkpoint.
    """
    variables_to_restore = tf.model_variables()
    assignment_map = { v.op.name: v for v in variables_to_restore }

    if ignore_missing:
        ckpt = tf.train.load_checkpoint(ckpt_to_start_from)
        existing_variables = ckpt.get_variable_to_shape_map()

        for variable_name in list(assignment_map.keys()):
            if variable_name not in existing_variables:
                del assignment_map[variable_name]
                tf.logging.info('Variable %s not found in checkpoint.', variable_name)

    tf.train.init_from_checkpoint(ckpt_to_start_from, assignment_map)


def make_metrics(logits, labels):
    metrics = {}
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)

    accuracy = tf.metrics.accuracy(labels, predictions, name='accuracy')

    metrics['accuracy'] = accuracy

    recall_5 = tf.metrics.recall_at_k(tf.expand_dims(tf.to_int64(labels), 1), logits, 5, name='recall_5')
    metrics['recall_5'] = recall_5

    return metrics


def _get_optimizer(optimizer, learning_rate):
    if optimizer is None:
        return tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)

    if callable(optimizer):
        return optimizer(learning_rate)

    return optimizer


def _get_value(params, name):
    if not isinstance(params, dict):
        params = vars(params)

    return params.get(name, None)


def _default_train_op_fn(loss, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = _get_value(params, 'learning_rate')
    optimizer = _get_optimizer(_get_value(params, 'optimizer'), learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def _call_network_fn(network_fn, images, is_training):
    sig = inspect.signature(network_fn)

    if isinstance(network_fn, partial):
        # Special case when the is_training keyword is bound by the
        # partial call already.
        if 'is_training' in network_fn.keywords:
            return network_fn(images)

    if 'is_training' in sig.parameters:
        return network_fn(images, is_training=is_training)
    else:
        return network_fn(images)


def make_model_fn(network_fn, train_op_fn):
    """ Create a model function by assembling the different parts of the model from
    the specified functions.

    Parameters
    ----------
    network_fn: A function that given the images and labels, constructs the
        main network structure.
    train_op_fn: A function that given the existing endpoints, constructs
        the training operation.

    Returns
    -------
    model_fn: A model_fn compatible with the `tf.Estimator` API.
    """
    if train_op_fn is None:
        train_op_fn = _default_train_op_fn

    def model_fn(features, labels, mode, params):
        # This dictionary contains all endpoints of interest in the network
        # and is propagated to the different functions.

        images = features['images']
        labels = tf.to_int32(labels)

        logits = _call_network_fn(network_fn, images, mode == tf.estimator.ModeKeys.TRAIN)
        predictions = tf.argmax(logits, 1, name='predictions', output_type=labels.dtype)

        batch_accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels), name='batch_accuracy'))

        tf.summary.scalar('accuracy', batch_accuracy, family='predictions')

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'logits': logits,
                             'top': predictions})

        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        tf.summary.scalar('model_loss', loss)
        eval_metrics = make_metrics(logits, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                eval_metric_ops=eval_metrics,
                loss=loss
            )

        optimizer_op = train_op_fn(loss, params)

        update_ops = list(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        update_ops.append(optimizer_op)

        with tf.control_dependencies(update_ops):
            train_op = tf.identity(loss, name='train_op')

        if _get_value(params, 'warm_start'):
            warm_start(_get_value(params, 'warm_start'), ignore_missing=True)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=eval_metrics,
            train_op=train_op,
            loss=loss)

    return model_fn
