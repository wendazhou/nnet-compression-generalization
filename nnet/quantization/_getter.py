""" Module to implement linear quantization of variables.

This module mostly implements quantization into a linear sequence between two values.
We also consider the case where we encode the sign separately and linearly encode
the magnitude only.

"""

import tensorflow as tf
import os
import inspect


QUANTIZED_VARIABLE_COLLECTION = 'quantized_variable'


def _evaluate_fn(fn, variable):
    if callable(fn):
        return fn(variable)
    else:
        return fn


def _add_to_collections(quantized_variable, quantized_variable_collections):
    if quantized_variable_collections is not None:
        for collection in quantized_variable_collections:
            tf.add_to_collection(collection, quantized_variable)


def _make_data(fn, config, collections):
    sig = inspect.signature(fn)

    if len(sig.parameters) > 1:
        return fn(config, collections)
    else:
        return fn(config)


def apply_quantization(variable, quantization,
                       quantized_variable_collections=None,
                       data_variable_collections=None,
                       init_collections=None,
                       name=None):
    """Obtains a quantized version of the variable using the specified quantization
    strategy.

    Parameters
    ----------
    variable: The variable to quantize.
    quantization: A descriptor of the quantization strategy to apply.
    quantized_variable_collections: A list of collections to which the quantized output value should be added.
    data_variable_collections: A list of collections to which to the data variables should be added.
    init_collections: A list of collections to which the init operations should be added.
    name: The name of the quantization operation

    Returns
    -------
    quantized_variable: A `tf.Tensor` representing the quantized value as a floating point number.
    init: A `tf.Operation` initializing the min and max variables according to the percentile of
        the original variables.
    """
    if quantized_variable_collections is None:
        quantized_variable_collections = [QUANTIZED_VARIABLE_COLLECTION]

    config = _evaluate_fn(quantization['config'], variable)

    with tf.variable_scope(name, 'quantization'):
        data = quantization['data'](config, data_variable_collections)

        with tf.name_scope('init'):
            init = quantization['init'](config, data)

        quantized_variable = quantization['quantization'](config, data)

    with tf.name_scope(variable.op.name + '/'):
        # Add the corresponding quantized variable under the same scope as the original
        # variable, so that we can easily find it in other operations.
        quantized_variable = tf.identity(quantized_variable, name='quantized')

    _add_to_collections(quantized_variable, quantized_variable_collections)
    _add_to_collections(init, init_collections)

    return quantized_variable, init


def _default_filter(*_, **kwargs):
    collections = kwargs.get('collections', None)
    return collections is not None and tf.GraphKeys.WEIGHTS in collections


def quantized_variable_getter(quantization,
                              variable_filter=None,
                              quantized_variable_collections=None,
                              data_variable_collections=None,
                              init_collections=None):
    """ Create a custom getter

    Parameters
    ----------
    quantization: The quantization strategy to apply.
    variable_filter: A function to indicate which variables to quantize. By default,
        only quantize variables in the `tf.GraphKeys.WEIGHTS` collection.
    quantized_variable_collections: An optional list of collections into which to add
        the output quantized variable.
    data_variable_collections: An optional list of collections into which to add the
        data variables created for the quantization.
    init_collections: An optional list of collections into which to add the init
        operations created for the quantization.

    Returns
    -------
    A custom getter which may be specified in a variable scope or a get_variable call.
    """
    if variable_filter is None:
        variable_filter = _default_filter

    def custom_getter(getter, *args, **kwargs):
        variable = getter(*args, **kwargs)

        if not variable_filter(*args, **kwargs):
            return variable

        name = os.path.basename(variable.op.name)

        quantized_variable, _ = apply_quantization(
            variable, quantization,
            quantized_variable_collections,
            data_variable_collections,
            init_collections,
            name=name + '/quantized')
        return quantized_variable

    return custom_getter


def assign_quantized_values_to_weights(variables=None, name=None,
                                       quantized_variable_collection=None) -> tf.Operation:
    """ Makes a tensorflow operation which assigns the quantized value to the
    weights.

    Parameters
    ----------
    variables: The list of variables for which to assign their masked value to.
    name: The name of the operation to create.
    quantized_variable_collection: The name of the collection from which to get
        the quantized variable.

    Returns
    -------
    A `tf.Operation` which assigns all the masked values.
    """
    if variables is None:
        variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    elif callable(variables):
        variables = variables()

    if quantized_variable_collection is None:
        quantized_variable_collection = QUANTIZED_VARIABLE_COLLECTION

    assign_ops = []

    with tf.name_scope(name, 'AssignQuantizedValues'):
        for v in variables:
            qvs = tf.get_collection(quantized_variable_collection, v.op.name)

            if len(qvs) != 1:
                tf.logging.warn('Skipped creating assignment for variable {0}'.format(v.name))
                continue

            mv = qvs[0]
            assign = tf.assign(v, mv, name='assign_qv')
            assign_ops.append(assign)

        assign = tf.group(*assign_ops, name='assign_all_qv')

    return assign


class CommitQuantizedValueHook(tf.train.CheckpointSaverListener):
    """  This class implements a CheckpointSaverListener which forces the original variables
    to take their quantized values at the given step (or steps).

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
        self._assign_op = assign_quantized_values_to_weights(self._variables_fn)

    def before_save(self, session, global_step_value):
        if global_step_value == self._last_step:
            tf.logging.info('Assigning quantized values to original weight variables before saving.')
            session.run(self._assign_op)
