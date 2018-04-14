""" Dynamic Network Surgery

Implementation of the ideas from:

Dynamic Network Surgery for Efficient DNNs
https://arxiv.org/pdf/1608.04493.pdf

"""

import tensorflow as tf
from . import MASK_COLLECTION, MASKED_WEIGHT_COLLECTION


def default_thresh_fn(scale):
    """ Returns a default thresholding function which considers
    the mean and standard deviation of the absolute values of the
    random variables.

    Parameters
    ----------
    scale: the quantity by which to scale the standard deviation.

    Returns
    -------
    fn: A function which computes the threshold.
    """
    def fn(variable, mask):
        absvar = tf.abs(variable, name='absvar')
        mean, var = tf.nn.moments(mask * absvar, axes=list(range(len(variable.shape))))
        std = tf.sqrt(var, name='std')

        thresh = tf.maximum(mean + scale * std, tf.zeros(shape=()), name='thresh')

        thresh_upper = tf.multiply(1.1, thresh, name='thresh_lower')
        thresh_lower = tf.multiply(0.9, thresh, name='thresh_lower')

        return thresh_lower, thresh_upper

    return fn


def _evaluate_function(fn_or_value, variable):
    if callable(fn_or_value):
        return fn_or_value(variable)
    else:
        return fn_or_value


def percentile_thresh_fn(target_sparsity, target_iterations,
                         update_steps=2000,
                         thresh_lower_scale=0.9,
                         thresh_upper_scale=1.1):
    """ Create a threshold function which thresholds based on a fixed target sparsity.

    Parameters
    ----------
    target_sparsity: The target sparsity for the given variable.
    target_iterations: The global step iteration count at which the sparsity should be reached.
    update_steps: The number of steps between each threshold updates.
    thresh_lower_scale: An adjustment to apply to the threshold to obtain the DNS lower bound.
    thresh_upper_scale: An adjustment to apply to the threshold to obtain the DNS upper bound.

    Returns
    -------
    fn: A function which computes the threshold for the given variable.
    """
    def fn(variable, mask):
        variable_target_sparsity = _evaluate_function(target_sparsity, variable)
        variable_target_iterations = _evaluate_function(target_iterations, variable)

        global_step = tf.train.get_or_create_global_step()
        sparsity_scale = (1 - tf.minimum(global_step, variable_target_iterations) / variable_target_iterations) ** 3

        sparsity = variable_target_sparsity * (1 - sparsity_scale)

        absvar = tf.abs(variable)
        thresh = tf.contrib.distributions.percentile(absvar, sparsity * 100)

        if update_steps > 1:
            # Store the computed threshold.
            stored_thresh = tf.get_variable('threshold_{0}'.format(variable.op.name),
                                            shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer(),
                                            trainable=False)

            store_thresh_every_n = tf.cond(
                tf.equal(tf.mod(global_step, update_steps), 0),
                lambda: tf.assign(stored_thresh, thresh),
                lambda: stored_thresh)
            thresh = store_thresh_every_n

        return thresh * thresh_lower_scale, thresh * thresh_upper_scale
    return fn


def _compute_dns_thresh(thresh_fn_or_scale, variable: tf.Variable, mask, name=None):
    dtype = mask.dtype

    if not callable(thresh_fn_or_scale):
        thresh_fn_or_scale = default_thresh_fn(thresh_fn_or_scale)

    with tf.name_scope(name, 'DynamicNetworkSurgeryThresh'):
        absvar = tf.abs(variable, name='absvar')

        thresh_lower, thresh_upper = thresh_fn_or_scale(variable, mask)

        to_prune = tf.less_equal(absvar, thresh_lower, name='to_prune')
        to_splice = tf.greater_equal(absvar, thresh_upper, name='to_splice')

        to_prune_f = tf.cast(to_prune, dtype=dtype)
        to_splice_f = tf.cast(to_splice, dtype=dtype)

        new_mask = tf.minimum(mask * (1 - to_prune_f) + to_splice_f, 1, name='new_mask')

        num_pruned = tf.count_nonzero(to_prune_f * mask, dtype=tf.int32, name='num_pruned')
        num_spliced = tf.count_nonzero(to_splice_f * (1 - mask), dtype=tf.int32, name='num_spliced')

        tf.summary.scalar('threshold_upper', thresh_upper)
        tf.summary.scalar('threshold_lower', thresh_lower)

    return new_mask, num_pruned, num_spliced


def _make_void(op):
    with tf.control_dependencies([op]):
        return tf.no_op('void')


def dns_thresh_op(scale, prob_thresh=None, variables=None, name=None):
    if variables is None:
        variables = tf.trainable_variables()

    all_thresh_ops = []
    all_num_pruned = []
    all_num_spliced = []
    count_thresh = []

    with tf.name_scope(name, 'DynamicNetworkSurgeryThreshold'):
        for v in variables:
            mc = tf.get_collection(MASK_COLLECTION, v.op.name)

            if len(tf.get_collection(MASK_COLLECTION, v.op.name)) != 1:
                continue

            mask = mc[0]

            new_mask, num_pruned, num_spliced = _compute_dns_thresh(scale, v, mask, name=v.op.name)

            should_thresh = tf.less_equal(tf.random_uniform((), dtype=tf.float32), prob_thresh,
                                          name='should_thresh')

            assign_op = tf.cond(
                should_thresh,
                lambda: _make_void(tf.assign(mask, new_mask)),
                lambda: tf.no_op(name='no_thresh')
            )

            all_thresh_ops.append(assign_op)
            all_num_pruned.append(num_pruned)
            all_num_spliced.append(num_spliced)
            count_thresh.append(tf.cast(should_thresh, dtype=tf.int32))

        total_to_prune = tf.add_n(all_num_pruned, name='total_to_prune')
        total_to_splice = tf.add_n(all_num_spliced, name='total_to_splice')

        total_pruned = tf.add_n([c * n for c, n in zip(count_thresh, all_num_pruned)])
        total_spliced = tf.add_n([c * n for c, n in zip(count_thresh, all_num_spliced)])

        count_thresh = tf.add_n(count_thresh, name='count_thresh')
        thresh = tf.group(all_thresh_ops, name='threshold_op')

        tf.summary.scalar('total_to_prune', total_to_prune)
        tf.summary.scalar('total_to_splice', total_to_splice)
        tf.summary.scalar('num_thresh_executed', count_thresh)
        tf.summary.scalar('total_pruned', total_pruned)
        tf.summary.scalar('total_spliced', total_spliced)
        tf.summary.scalar('thresh_prob', prob_thresh)

    return thresh


def dns_grad_op(loss, optimizer: tf.train.Optimizer, variables=None, global_step=None):
    """ Create an operation the updates the weights by gradient descent.

    In DNS, the weights are updated according to their derivative with respect to the masked
    values, but the update is applied to the non-masked values, so that zeroed-out weights may
    still change and in particular be spliced back in if necessary.

    Parameters
    ----------
    loss: A `tf.Tensor` representing the loss.
    optimizer: The optimizer to use.
    variables: The variables for which to create the gradient operation.
    global_step: An optional global step to increment.

    Returns
    -------
    train_op: An tensorflow op that when run updates the variables according to the gradient.
    """
    if variables is None:
        variables = tf.trainable_variables()

    replaced = {}

    wrt_variables = []

    num_replaced = 0

    for v in variables:
        # look for variables having shadow values.
        mvs = tf.get_collection(MASKED_WEIGHT_COLLECTION, v.op.name)

        if len(mvs) == 0:
            wrt_variables.append(v)
        elif len(mvs) == 1:
            num_replaced += 1
            wrt_variables.append(mvs[0])
            replaced[mvs[0]] = v
        else:
            raise ValueError('More than one masked weight for a given variable.')

    tf.logging.info('Replaced {0} variables for Dynamic Network Surgery'.format(num_replaced))

    grads_and_vars = optimizer.compute_gradients(loss, wrt_variables)
    grads_and_vars = [(g, replaced.get(v, v)) for g, v in grads_and_vars]

    train_op = optimizer.apply_gradients(grads_and_vars, global_step, 'dns_grad_op')

    return train_op


def make_dns_train_op(loss, prob_thresh=None,
                      optimizer=None, thresh_fn_or_scale=None, variables=None,
                      global_step=None):
    """ Make a Dynamic Network Surgery training op.

    Parameters
    ----------
    loss: The loss to be optimized.
    prob_thresh: The probability with which to apply the thresholding.
    optimizer: The optimizer to use.
    thresh_fn_or_scale: Either a function that computes the thresholding values,
        or a scalar that control the thresholding scale.
    variables: The set of variables to threshold.
    global_step: A variable representing the global step.

    Returns
    -------
    An `tf.Operation`, which when run, both trains and updates the masks.
    """
    if prob_thresh is None:
        prob_thresh = tf.train.exponential_decay(1.0, global_step, 2000, 0.9)

    if optimizer is None:
        optimizer = tf.train.AdamOptimizer()

    if thresh_fn_or_scale is None:
        thresh_fn_or_scale = tf.constant(4.0, dtype=tf.float32, name='dns_thresh_scale')

    grad_op = dns_grad_op(loss, optimizer, variables, global_step=global_step)

    with tf.control_dependencies([grad_op]):
        thresh_op = dns_thresh_op(thresh_fn_or_scale, prob_thresh, variables)

    return thresh_op
