import tensorflow as tf
import os
import numpy as np
import pathlib
import types
import random
import time


class HGEST(object):

    '''
    Human gestures dataset.

    Parameters
    ----------
    key : str e.g. 'raw' or 'normalized'. 
        Identifies the version of the dataset to use.

    Attributes
    ----------
    files : Generator[Path].
        One file per subject containing the samples that make up the dataset.
    '''

    def __init__(self, key='raw'):
        fname = 'hgest.tar.gz'
        origin = 'https://storage.googleapis.com/exoskelebox/hgest.tar.gz'
        path: str = tf.keras.utils.get_file(
            fname, origin, extract=True, file_hash='452ec3c4e89eb187825725092f9c6a8e')
        path = path.rsplit('.', 2)[0]
        path = pathlib.Path(os.path.join(path, key))

        self._path = path

    def _parse(self, example_proto):
        # Parse the input `tf.Example` proto using the feature description dictionary.
        parsed_example = tf.io.parse_single_example(
            example_proto, self.feature_description)
        return parsed_example, parsed_example.pop('label')

    def _dataset(self, file_or_files):
        """
        Default dataset constructor
        """
        dataset: tf.data.Dataset = None

        if isinstance(file_or_files, (list, types.GeneratorType)):
            dataset = tf.data.Dataset\
                .from_tensor_slices(list(map(str, file_or_files)))\
                .interleave(
                    tf.data.TFRecordDataset,
                    cycle_length=tf.data.experimental.AUTOTUNE,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(str(file_or_files))

        return dataset\
            .map(
                map_func=self._parse,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

    @property
    def feature_description(self):
        return {
            'subject_id': tf.io.FixedLenFeature([], tf.int64),
            'gesture': tf.io.FixedLenFeature([], tf.string),
            'repetition': tf.io.FixedLenFeature([], tf.int64),
            'readings': tf.io.FixedLenFeature([15], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

    @property
    def files(self):
        return self._path.glob('*.tfrecord')

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def dataset(self):
        return self._dataset(self.files)

    def feature_layer(self, cols: [] = None) -> tf.keras.layers.Dense:
        valid_cols = self.feature_description.keys()

        # Label is not a valid column for the feature layer
        valid_cols.remove('label')

        # Ensure unique columns
        cols = set(cols)

        if not cols:
            cols = valid_cols

        if not set(cols).issubset(valid_cols):
            exit(
                f"One or more invalid feature column names, valid feature column names are {', '.join(valid_cols)}.")

        return tf.keras.layers.DenseFeatures([self.feature_description[col](col) for col in cols])

    def subjects(self, shuffle=True):
        files = list(self.files)

        if shuffle:
            random.shuffle(files)

        for file in files:
            yield self._dataset(file)\
                .shuffle(2 ** 14, reshuffle_each_iteration=False)

    def k_fold(self, shuffle=True, batch_size=64):
        repetitions = None

        def is_test(i, x):
            return i % 2 == 0

        def is_val(i, x):
            return not is_test(i, x)

        def deenumerate(i, x):
            return x

        def repetition(x, y):
            return x['repetition']

        for subject in self.subjects(shuffle):
            # Fetch the unique repetition feature values
            if not repetitions:
                repetitions = subject\
                    .map(repetition)\
                    .apply(tf.data.experimental.unique())

            for rep in repetitions:

                def is_test_rep(x, y):
                    return x['repetition'] == rep

                def is_train(x, y):
                    return not is_test_rep(x, y)

                train_dataset = subject\
                    .filter(is_train)

                test_dataset = subject\
                    .filter(is_test_rep)

                val_dataset = test_dataset\
                    .enumerate()\
                    .filter(is_val)\
                    .map(
                        deenumerate,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

                test_dataset = test_dataset\
                    .enumerate()\
                    .filter(is_test)\
                    .map(
                        deenumerate,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

                train_dataset = train_dataset\
                    .cache()\
                    .shuffle(2 ** 14)\
                    .batch(
                        batch_size=batch_size,
                        drop_remainder=True)\
                    .prefetch(tf.data.experimental.AUTOTUNE)

                val_dataset = val_dataset\
                    .cache()\
                    .shuffle(2 ** 14)\
                    .batch(
                        batch_size=batch_size,
                        drop_remainder=True)\
                    .prefetch(tf.data.experimental.AUTOTUNE)

                test_dataset = test_dataset\
                    .cache()\
                    .shuffle(2 ** 14, reshuffle_each_iteration=False)\
                    .batch(
                        batch_size=batch_size,
                        drop_remainder=True)\
                    .prefetch(tf.data.experimental.AUTOTUNE)

                yield train_dataset, val_dataset, test_dataset
