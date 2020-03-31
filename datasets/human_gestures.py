from __future__ import absolute_import

from typing import List, Tuple
import tensorflow as tf
import time
from collections import defaultdict
import glob
import random

feature_description = {
    'subject_id': tf.io.FixedLenFeature([], tf.int64),
    'subject_gender': tf.io.FixedLenFeature([], tf.string),
    'subject_age': tf.io.FixedLenFeature([], tf.int64),
    'subject_fitness': tf.io.FixedLenFeature([], tf.int64),
    'subject_handedness': tf.io.FixedLenFeature([], tf.string),
    'subject_impairment': tf.io.FixedLenFeature([], tf.string),
    'subject_wrist_circumference': tf.io.FixedLenFeature([], tf.float32),
    'subject_forearm_circumference': tf.io.FixedLenFeature([], tf.float32),
    'gesture': tf.io.FixedLenFeature([], tf.string),
    'repetition': tf.io.FixedLenFeature([], tf.int64),
    'reading_count': tf.io.FixedLenFeature([], tf.int64),
    'readings': tf.io.FixedLenFeature([15], tf.int64),
    'arm_calibration_iterations': tf.io.FixedLenFeature([], tf.int64),
    'arm_calibration_values': tf.io.FixedLenFeature([8], tf.int64),
    'wrist_calibration_iterations': tf.io.FixedLenFeature([], tf.int64),
    'wrist_calibration_values': tf.io.FixedLenFeature([7], tf.int64),
    'timedelta': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the feature description dictionary.
    parsed_example = tf.io.parse_single_example(
        example_proto, feature_description)
    return parsed_example, parsed_example.pop('label')


def _numeric_column(shape=(1,), default_value=None, dtype=tf.dtypes.float32, normalizer_fn=None):
    return lambda key: tf.feature_column.numeric_column(
        key,
        shape=shape,
        default_value=default_value,
        dtype=dtype,
        normalizer_fn=normalizer_fn)


def _indicator_column(vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0):
    return lambda key: tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key,
            vocabulary_list,
            dtype=dtype,
            default_value=default_value,
            num_oov_buckets=num_oov_buckets))


def _bucketized_column(boundaries, default_value=None, dtype=tf.dtypes.float32, normalizer_fn=None):
    return lambda key: tf.feature_column.bucketized_column(
        _numeric_column(
            default_value=default_value,
            dtype=dtype,
            normalizer_fn=normalizer_fn)(key),
        boundaries)


def get_feature_layer(cols: [] = None) -> tf.keras.layers.Dense:

    feature_columns = {
        'subject_id': _numeric_column(dtype=tf.uint16),
        'subject_gender': _indicator_column(['m', 'f']),
        'subject_age': _bucketized_column([18, 25, 30, 35, 40, 45, 50, 55, 60, 65], dtype=tf.uint8),
        'subject_fitness': _bucketized_column([2, 4, 6, 8], dtype=tf.uint8),
        'subject_handedness': _indicator_column(['l', 'r', 'a']),
        'subject_impairment': _indicator_column(['t', 'f']),
        'subject_wrist_circumference': _numeric_column(dtype=tf.float32),
        'subject_forearm_circumference': _numeric_column(dtype=tf.float32),
        'repetition': _numeric_column(dtype=tf.uint16),
        'reading_count': _numeric_column(dtype=tf.uint32),
        'readings':  _numeric_column(dtype=tf.dtypes.uint8, shape=15),
        'arm_calibration_iterations': _numeric_column(dtype=tf.uint16),
        'arm_calibration_values': _numeric_column(dtype=tf.dtypes.uint8, shape=8),
        'wrist_calibration_iterations': _numeric_column(dtype=tf.uint16),
        'wrist_calibration_values': _numeric_column(dtype=tf.dtypes.uint8, shape=7),
        'timedelta': _numeric_column(dtype=tf.uint32),
    }

    if not cols:
        cols = feature_columns.keys()

    if not set(cols).issubset(feature_columns.keys()):
        exit(
            f"One or more invalid feature column names, valid feature column names are {', '.join(feature_columns.keys())}.")

    return tf.keras.layers.DenseFeatures([feature_columns[col](col) for col in cols])


def get_data(shuffle=True, test=0.2):
    """
    Retreives the human gestures dataset.
    """
    fname = 'hgest.tar.gz'
    origin = 'https://storage.googleapis.com/exoskelebox/hgest.tar.gz'
    path: str = tf.keras.utils.get_file(fname, origin, extract=True)
    path = path.rsplit('.', 2)[0]
    files = glob.glob(path + '/*.tfrecord')

    if shuffle:
        random.shuffle(files)

    num_test = int(test * len(files))
    train_files, test_files = files[num_test:], files[:num_test]

    train = tf.data.TFRecordDataset(
        filenames=train_files,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    """ .prefetch(
        tf.data.experimental.AUTOTUNE
    ) """

    test = tf.data.TFRecordDataset(
        filenames=test_files,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).map(
        _parse_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    """ .prefetch(
        tf.data.experimental.AUTOTUNE
    ) """

    return train, test


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)
