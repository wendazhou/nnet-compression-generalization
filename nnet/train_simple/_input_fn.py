from tensorflow.contrib.data import map_and_batch


def make_input_fn(dataset_factory,
                  preprocessing_fn,
                  n_epochs=None,
                  image_size=32,
                  batch_size=64,
                  shuffle=True):
    """Create an input function.

    Parameters
    ----------
    dataset_factory: The dataset to create the input from, or a function that creates
        the dataset.
    preprocessing_fn: The preprocessing function to apply
    n_epochs: The number of epochs to run through the data,
        or None to loop endlessly.
    image_size: The size of the image (along a single side)
    batch_size: Minibatch size to use.

    Returns
    -------
    input_fn: A input function which is compatible with the estimator API.
    """
    def _preprocess_map(record):
        image = preprocessing_fn(record, image_size, image_size)
        return {'images': image}, record['image/class/label']

    def input_fn():
        ds = dataset_factory()

        if shuffle:
            ds = ds.shuffle(1000, reshuffle_each_iteration=True)

        ds = ds.repeat(n_epochs)

        ds = ds.apply(map_and_batch(_preprocess_map, batch_size, 4))

        ds = ds.prefetch(16)
        return ds

    return input_fn
