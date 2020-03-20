from __future__ import absolute_import

from database import Database
from typing import List, Tuple
import tensorflow as tf
from collections import defaultdict


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


# the columns present in the database that can be used as a feature
# each feature should be associated with a feature column constructor function
FEATURE_COLUMNS = {
    'subject_gender': _indicator_column(['m', 'f']),
    'subject_age': _bucketized_column([18, 25, 30, 35, 40, 45, 50, 55, 60, 65], dtype=tf.uint8),
    'subject_fitness': _bucketized_column([2, 4, 6, 8], dtype=tf.uint8),
    'subject_handedness': _indicator_column(['l', 'r', 'a']),
    'subject_impairment': _indicator_column(['True', 'False']),
    'subject_wrist_circumference': _numeric_column(),
    'subject_forearm_circumference': _numeric_column(),
    'repetition': _numeric_column(),
    'timestamp': None,  # TODO
    'readings': _numeric_column(dtype=tf.dtypes.uint8, shape=15),
    'arm_calibration_iterations': _numeric_column(dtype=tf.dtypes.uint16),
    'arm_calibration_values': _numeric_column(dtype=tf.dtypes.uint8, shape=8),
    'wrist_calibration_iterations': _numeric_column(dtype=tf.dtypes.uint16),
    'wrist_calibration_values': _numeric_column(dtype=tf.dtypes.uint8, shape=7),
}


def _categorical(data: []):
    numerical_map = {x: i for i, x in enumerate(set(data))}
    numerical_data = [numerical_map[x] for x in data]
    return tf.keras.utils.to_categorical(numerical_data)


# the columns present in the database that can be used as a label
# each label can be associated with a preprocessing function
LABEL_COLUMNS = {
    'gesture': _categorical
}


def get_feature_layer(dataset: tf.data.Dataset) -> tf.keras.layers.Dense:
    feature_columns = [FEATURE_COLUMNS[key](
        key) for key in dataset.element_spec[0].keys()]
    return tf.keras.layers.DenseFeatures(feature_columns)


def get_data(fcols: [], lcol: str) -> tf.data.Dataset:
    """
    Retreives data from database.
    The dataset should be batched before attempting to use it as input for a model.
    """

    # validate feature- and label columns
    if not set(fcols).issubset(FEATURE_COLUMNS.keys()):
        exit(
            f"One or more invalid feature columns, valid feature columns are {', '.join(FEATURE_COLUMNS)}.")

    if lcol not in LABEL_COLUMNS.keys():
        exit(
            f"Label column invalid, valid label columns are {', '.join(LABEL_COLUMNS)}.")

    # combine the column names
    colnames = set([*fcols, lcol])

    # query the database for the columns
    cur = Database().query(
        f"SELECT {','.join(colnames)} FROM training")

    # construct a data dictionary
    data_dict = defaultdict(list)
    [[data_dict[feature].append(
        row[index]) for index, feature in enumerate(colnames)] for row in cur]
    data_dict = dict(data_dict)

    # pop the labels
    labels = data_dict.pop(lcol)

    # preprocess the labels if preprocessing func exists
    if LABEL_COLUMNS[lcol] is not None:
        labels = LABEL_COLUMNS[lcol](labels)

    # construct dataset
    dataset = tf.data.Dataset.from_tensor_slices((data_dict, labels))

    return dataset


get_data(['readings'], 'gesture')
