import tensorflow as tf
import os
import numpy as np
import pathlib
import types


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

    def __init__(self, key='normalized'):
        fname = 'hgest.tar.gz'
        origin = 'https://storage.googleapis.com/exoskelebox/hgest.tar.gz'
        path: str = tf.keras.utils.get_file(
            fname, origin, extract=True, file_hash='ba69480e37c725eab4676f95d44b0720')
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
            )\
            .cache()

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
        return self._dataset(next(self.files))

    def feature_layer(self, cols: [] = None) -> tf.keras.layers.Dense:
        valid_cols = self.feature_description.keys()

        if not cols:
            cols = valid_cols

        if not set(cols).issubset(valid_cols):
            exit(
                f"One or more invalid feature column names, valid feature column names are {', '.join(valid_cols)}.")

        return tf.keras.layers.DenseFeatures([self.feature_description[col](col) for col in cols])


hgest = HGEST()
print(hgest.dataset)
