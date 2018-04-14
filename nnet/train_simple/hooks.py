""" Utility hooks for training. """


import tensorflow as tf


class ExecuteAtStepHook(tf.train.SessionRunHook):
    """ This hook executes the given operation at the given steps. """
    def __init__(self, steps, op_fn, name=None):
        """ Creates a hook at that executes the operation created by `op_fn` at
        the given `steps`.

        Parameters
        ----------
        steps: A list of steps at which to execute the operation, or a predicated
            indicating whether to execute given the step number.
        op_fn: A function that creates the operation to run.
        """
        self._step = 0

        if callable(steps):
            self._trigger_steps = steps
        else:
            self._trigger_steps = set(steps)

        self._op_fn = op_fn
        self._op = None
        self._name = name

    def begin(self):
        self._op = self._op_fn()

    def before_run(self, run_context):
        if self._step in self._trigger_steps:
            tf.logging.info('Executing hook: {0}'.format(self._name))
            return tf.train.SessionRunArgs(self._op)

    def after_run(self, run_context, run_values):
        self._step += 1


class ExecuteAtSessionCreateHook(tf.train.SessionRunHook):
    """ This hook executes the given operation after the session is created.

    The hook also checks the global step to ensure that it is only executed
    at the start of training and not when a session is restored during training.

    """
    def __init__(self, op_fn, name=None):
        """ Creates a new hook which executes the operation returned by `op_fn`.

        Parameters
        ----------
        op_fn: A function which creates the operation to execute.
        name: The name of the hook (used for logging purposes).
        """
        self._name = name
        self._op_fn = op_fn
        self._op = None
        self._global_step = None

    def begin(self):
        self._op = self._op_fn()
        self._global_step = tf.train.get_or_create_global_step()

    def after_create_session(self, session, coord):
        if session.run(self._global_step) != 0:
            return

        if self._name is not None:
            tf.logging.info('Executing hook: {0}'.format(self._name))

        session.run(self._op)
