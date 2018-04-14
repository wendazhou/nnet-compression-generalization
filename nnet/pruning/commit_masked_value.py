""" Utilities to commit masked values to the weights

"""

import tensorflow as tf
import numpy as np


def commit_values(in_ckpt, out_ckpt):
    """ Commits the masked values from the input checkpoint to the output checkpoint.

    Parameters
    ----------
    in_ckpt: The input checkpoint from which to read the values.
    out_ckpt: The output checkpoint to which to save the values.
    """
    ckpt = tf.train.load_checkpoint(in_ckpt)

    variable_shapes = ckpt.get_variable_to_shape_map()
    variable_types = ckpt.get_variable_to_dtype_map()

    with tf.Graph().as_default():
        variables = []

        for variable_name, variable_type in variable_types.items():
            mask_name = variable_name + '/mask'
            value = ckpt.get_tensor(variable_name)

            if mask_name in variable_types:
                value = np.multiply(value, ckpt.get_tensor(mask_name))

            var = tf.Variable(value,
                              dtype=variable_type,
                              expected_shape=variable_shapes[variable_name],
                              name=variable_name)

            variables.append(var)

        saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = ""

        with tf.Session(config=config) as session:
            session.run(init)
            saver.save(session, out_ckpt, write_meta_graph=False)


def _main():
    import argparse

    parser = argparse.ArgumentParser(prog='commit_masked_values',
                                     description='Merge masked values into weights')

    parser.add_argument('-i', '--input', type=str, help='Input checkpoint path')
    parser.add_argument('-o', '--output', type=str, help='Output checkpoint path')

    parsed = parser.parse_args()

    if parsed.input is None or parsed.output is None:
        parser.print_usage()
        return

    commit_values(parsed.input, parsed.output)


if __name__ == '__main__':
    _main()
