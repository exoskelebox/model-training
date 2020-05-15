from __future__ import absolute_import
import tensorflow as tf
import time
from tqdm import tqdm


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


def split(dataset, ratios=(1, 1), offset=0):
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
