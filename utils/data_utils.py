from __future__ import absolute_import
import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import itertools


@tf.function
def deenumerate(i, x):
    return x


def exclude(key, val, dtype):
    """
    Use as a filter function to exclude batches that contain the specified key and value
    """
    @tf.function
    def pred(x, *args):
        x = x[key]
        y = tf.constant(val)

        if dtype:
            x = tf.cast(x, dtype)
            y = tf.cast(y, dtype)

        equal = tf.math.equal(x, y)
        return tf.math.reduce_any(equal)
    return pred


def split_old(dataset, ratios=(1, 1), offset=0):
    """
    Splits data according to provided ratios. Returns list of datasets.
    """
    ratio_sum = sum(ratios)
    subsets = []

    @tf.function
    def is_subset(i, offset, ratio):
        # 2*ratio_sum ensures offset does not cause it to reach negative numbers
        return (i + (2*ratio_sum) - offset) % ratio_sum < ratio

    dataset = dataset.enumerate()
    for ratio in ratios:
        @tf.function
        def pred(i, x): return is_subset(i, offset, ratio)

        subset = dataset\
            .filter(pred)\
            .map(deenumerate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        subsets.append(subset)
        offset += ratio

    return subsets


def split(dataset, ratios=(1, 1)):
    assert len(ratios) == 2

    @tf.function
    def flatten_nested(*ds):
        return ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds)

    r1, r2 = ratios

    d1 = dataset.window(r1, r1 + r2).flat_map(flatten_nested)
    d2 = dataset.skip(r1).window(r2, r1 + r2).flat_map(flatten_nested)
    return d1, d2


def shuffle(*datasets, buffer_size=2**14, seed=None, reshuffle_each_iteration=None):
    return [dataset.shuffle(buffer_size, seed, reshuffle_each_iteration) for dataset in datasets]


def negate(f):
    def g(*args, **kwargs):
        return not f(*args, **kwargs)
    return g


def key_val_split(dataset, key, val, dtype=None):
    """
    Splits data according to provided key and value.
    Returns two datasets, the second contains data matching the predicate and 
    the first contains the remainder.
    """

    pred = exclude(key, val, dtype)
    x = dataset.filter(pred)
    y = dataset.filter(negate(pred))

    return y, x


def feature_fold(dataset, key):
    """
    Produces a feature-fold generator based on given feature key.
    """

    def fn(x, *args):
        return x[key]

    unique_values = dataset\
        .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .apply(tf.data.experimental.unique())

    for val in unique_values:
        yield key_val_split(dataset, key, val)


def files_fold(file_paths):
    for i, current in enumerate(file_paths):
        rest = file_paths[:i] + file_paths[i+1:]


def benchmark(dataset):
    """
    Does two passes of the dataset to assess the execution time
    """
    start_time = time.perf_counter()
    sample_count = None
    for i in range(2):
        tmp = 0
        for sample in tqdm(dataset, total=sample_count):
            tmp += 1
        sample_count = tmp
    tf.print("Execution time:", time.perf_counter() - start_time)


def confusion_matrix(model, test_data, class_names=None):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """

    # Use the model to predict the values from the validation dataset.
    test_predictions_raw = model.predict(test_data)
    test_predictions = np.argmax(test_predictions_raw, axis=1)

    test_labels = []
    gestures = {}

    # Calculate the confusion matrix.
    cm = tf.math.confusion_matrix(test_labels, test_predictions).numpy()

    figure = plt.figure(figsize=(8, 8))
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))

    # Rotate the tick labels and set their alignment
    plt.xticks(tick_marks, class_names, rotation=45,
               ha="right", rotation_mode="anchor")
    plt.yticks(tick_marks, class_names, rotation=45,
               ha="right", rotation_mode="anchor")

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)
                   [:, np.newaxis], decimals=2)

    # Create colorbar
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlOrRd)
    plt.colorbar()

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure
