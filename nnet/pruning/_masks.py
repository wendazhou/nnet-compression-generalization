""" This module implements code to mask variables when pruning model. """

import tensorflow as tf
import numpy as np
import os

#: The key for the masks collection.
MASK_COLLECTION = 'mask'

#: The key for the masked weights collection.
MASKED_WEIGHT_COLLECTION = 'masked_weight'


def apply_mask(weights: tf.Variable,
               mask_collections=None,
               masked_weight_collections=None):
    """ Creates a mask for the given variable, and returns a masked version
    of the variable.

    Parameters
    ----------
    weights: The variable for which to create the masks.
    mask_collections: A list of collections into which to add the mask variable.
        By default, only adds to MASK_COLLECTION.
    masked_weight_collections: A list of collections into which to add the masked
        weights. By default, only add to MASKED_WEIGHT_COLLECTION.

    Returns
    -------
    masked_weight: The value of the masked variable.
    mask: The variable corresponding to the created mask.
    """
    variable_name = os.path.basename(weights.op.name)

    with tf.variable_scope(variable_name) as vs:
        mask = tf.get_variable(
            'mask',
            weights.get_shape(),
            initializer=tf.ones_initializer,
            trainable=False,
            dtype=weights.dtype
        )

        name_scope_name = vs.name

    # Need to re-open a name scope with an absolute name
    # as otherwise the variable_scope is set correctly
    # but the name_scope is not set correctly.
    with tf.name_scope(name_scope_name + '/'):
        masked_weight = tf.multiply(weights, mask, 'masked_weight')

    if mask_collections is not None:
        for collection in mask_collections:
            tf.add_to_collection(collection, mask)
    else:
        tf.add_to_collection(MASK_COLLECTION, mask)

    if masked_weight_collections is not None:
        for collection in masked_weight_collections:
            tf.add_to_collection(collection, masked_weight)
    else:
        tf.add_to_collection(MASKED_WEIGHT_COLLECTION, masked_weight)

    return masked_weight


def _default_mask_filter(*_, **kwargs):
    collections = kwargs.get('collections', None)
    return collections is not None and tf.GraphKeys.WEIGHTS in collections


def masked_variable_getter(mask_filter=None, mask_collections=None, masked_weight_collections=None):
    """ This function is a custom getter (compatible with `tf.get_variable`), which wraps the returned
        variable as a masked variable.

    Parameters
    ----------
    mask_filter: A function indicating which variables to mask. By default, only mask variables if
        they are to be added to the weights collection.
    mask_collections: A list of collections into which the mask variables should be added.
    masked_weight_collections: A list of collections into which the masked weights should be added.

    Returns
    -------
    custom_getter: A custom getter that wraps the specified variables as masked weights.
    """
    if mask_filter is None:
        mask_filter = _default_mask_filter

    def custom_getter(getter, *args, **kwargs):
        variable = getter(*args, **kwargs)

        if mask_filter(*args, **kwargs):
            return apply_mask(variable, mask_collections, masked_weight_collections)
        else:
            return variable

    return custom_getter


def make_pruning_summaries():
    """ This function creates summaries indicating the pruned proportion for all pruned
    weights, and the global pruned proportion.
    """

    masks = tf.get_collection(MASK_COLLECTION)

    masks_density = []
    masks_size = []

    with tf.name_scope('pruning_summaries'):
        for mask in masks:
            mask_density = tf.reduce_mean(mask, name=mask.op.name + '/density')
            masks_density.append(mask_density)
            masks_size.append(np.prod(mask.shape).value)
            tf.summary.scalar(mask.op.name, mask_density)

        ms = tf.constant(masks_size, dtype=tf.float32, name='masks_size')
        md = tf.stack(masks_density, name='masks_density')

        prune_amount = tf.subtract(1., tf.divide(tf.reduce_sum(ms * md), tf.reduce_sum(ms)), name='prune_amount')
    tf.summary.scalar('prune_amount', prune_amount)


def assign_masked_values_to_weights(variables=None, name=None) -> tf.Operation:
    """ Makes a tensorflow operation which assigns the masked value to the
    weights.

    Parameters
    ----------
    variables: The list of variables for which to assign their masked value to.
    name: The name of the operation to create.

    Returns
    -------
    A `tf.Operation` which assigns all the masked values.
    """
    if variables is None:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    assign_ops = []

    with tf.name_scope(name, 'AssignMaskedValues'):
        for v in variables:
            mvs = tf.get_collection(MASKED_WEIGHT_COLLECTION, v.op.name)

            if len(mvs) != 1:
                tf.logging.warn('Skipped creating assignment for variable {0}'.format(v.name))
                continue

            mv = mvs[0]
            assign = tf.assign(v, mv, name='assign_mv')
            assign_ops.append(assign)

        assign = tf.group(*assign_ops, name='assign_all_mv')

    return assign


class CommitMaskedValueHook(tf.train.CheckpointSaverListener):
    """  This class implements a CheckpointSaverListener which forces the original variables
    to take their truncated values at the given step (or steps).

    """
    def __init__(self, last_step, variables_fn=None):
        """ Initializes a new hook which assigns the masked values to the given variables.

        Parameters
        ----------
        last_step: The step at which to commit the variable values.
        variables_fn: A function that gets the variables to set to the assigned values.
        """
        self._last_step = last_step
        self._assign_op = None
        self._variables_fn = variables_fn

    def begin(self):
        if self._variables_fn is not None:
            variables = self._variables_fn()
        else:
            variables = None
        self._assign_op = assign_masked_values_to_weights(variables)

    def before_save(self, session, global_step_value):
        if global_step_value == self._last_step:
            tf.logging.info('Assigning masked values to original weight variables before saving.')
            session.run(self._assign_op)
