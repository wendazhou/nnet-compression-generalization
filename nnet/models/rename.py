""" Utilities for model files. """

import tensorflow as tf
import typing


def rename_variables(rename: typing.Callable[[str], str],
                     checkpoint: str,
                     output_checkpoint: str):
    """ Renames the variables in a given checkpoint

    Parameters
    ----------
    rename: the function to use to rename the variables.
    checkpoint: the checkpoint to read the variables from.
    output_checkpoint: the checkpoint to save the variables to.
    """
    tf.logging.info('Loading variables from checkpoint {0}'.format(checkpoint))
    ckpt = tf.train.load_checkpoint(checkpoint)

    ckpt_vars = ckpt.get_variable_to_shape_map()

    with tf.Graph().as_default():
        for v_name in ckpt_vars.keys():
            new_name = rename(v_name)
            value = ckpt.get_tensor(v_name)
            tf.Variable(value, name=new_name)

        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            tf.logging.info('Saving variables to checkpoint {0}'.format(output_checkpoint))
            saver.save(session, output_checkpoint)


def main():
    import argparse
    import sys
    import importlib
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    parser = argparse.ArgumentParser('rename', description='Renames variables in checkpoint file.')
    parser.add_argument('-i', '--i', help='input checkpoint')
    parser.add_argument('-o', '--o', help='output checkpoint')
    parser.add_argument('-m', '--model', help='model to rename with')

    args = parser.parse_args(sys.argv[1:])

    module_name, function_name = args.model.rsplit('.', 1)

    module = importlib.import_module(module_name, __package__)
    rename_fn = getattr(module, function_name)

    tf.logging.set_verbosity(tf.logging.INFO)
    rename_variables(rename_fn, args.i, args.o)


if __name__ == '__main__':
    main()
