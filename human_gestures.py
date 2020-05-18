import tensorflow as tf
import types
from pathlib import Path
from utils.data_utils import benchmark, feature_fold, key_val_split, split
from utils.iter_utils import fold
import os
from itertools import tee


class HumanGestures():
    def __init__(self, batch_size=64):
        self._batch_size = batch_size

    @staticmethod
    def path() -> Path:
        fname = 'normalized_hgest.tar.gz'
        origin = 'https://storage.googleapis.com/exoskelebox/normalized_hgest.tar.gz'
        path: str = tf.keras.utils.get_file(
            fname, origin, extract=True, file_hash='98ae04fe4d1d4ad6fbcf5da8f7f7f648')
        return Path(path.rsplit('.', 2)[0])

    @staticmethod
    def feature_description():
        """
        Feature description dictionary.
        """
        return {
            'subject_id': tf.io.FixedLenFeature([], tf.int64),
            'gesture': tf.io.FixedLenFeature([], tf.string),
            'repetition': tf.io.FixedLenFeature([], tf.int64),
            'readings': tf.io.FixedLenFeature([15], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

    @staticmethod
    def feature_inputs():
        """
        Feature input dictionary, used for the construction of models using the functional Keras API.
        """
        return {
            'repetition': tf.keras.Input((1,), dtype=tf.int64, name='repetition'),
            'readings': tf.keras.Input((15,), dtype=tf.float32, name='readings')
        }

    @staticmethod
    def feature_layer(cols: [] = None) -> tf.keras.layers.Dense:

        def _numeric_column(shape=(1,), default_value=None, dtype=tf.dtypes.float32, normalizer_fn=None):
            return lambda key: tf.feature_column.numeric_column(
                key,
                shape=shape,
                default_value=default_value,
                dtype=dtype,
                normalizer_fn=normalizer_fn)

        feature_columns = {
            'repetition': _numeric_column(dtype=tf.uint16),
            'readings':  _numeric_column(dtype=tf.float32, shape=15),
        }

        valid_cols = feature_columns.keys()

        if not cols:
            cols = valid_cols

        if not set(cols).issubset(valid_cols):
            exit(
                f"One or more invalid feature column names, valid feature column names are {', '.join(valid_cols)}.")

        return tf.keras.layers.DenseFeatures([feature_columns[col](col) for col in cols])

    @property
    def batch_size(self):
        return self._batch_size

    def _dataset(self, file_or_files):
        """
        Dataset constructor
        """
        dataset: tf.data.Dataset = None

        def parse_batch(example_protos):
            parsed_examples = tf.io.parse_example(
                example_protos, self.feature_description())
            return parsed_examples, parsed_examples.pop('label')

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
            .batch(
                batch_size=self.batch_size, drop_remainder=True)\
            .map(
                map_func=parse_batch,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()

    def dataset(self):
        """
        Returns a dataset with all the human gestures data
        """
        return self._dataset(self.path().glob('*.tfrecord'))

    def subject_datasets(self, fold_current=True, flatten_remainder=True):
        subject_paths = [Path(f.path)
                         for f in os.scandir(self.path()) if f.is_dir()]

        for csub, rsub in fold(subject_paths):
            csub_reps = list(csub.glob('*.tfrecord'))
            if flatten_remainder:
                rsub_reps = self._dataset([rep for sub in rsub for rep in sub.glob('*.tfrecord')])
            else:
                rsub_reps = [self._repetition_datasets([rep for rep in sub.glob('*.tfrecord')]) for sub in rsub]
            yield (self._repetition_datasets(csub_reps) if fold_current else self._dataset(csub_reps)), rsub_reps

    def _repetition_datasets(self, files):
        for crep, rrep in fold(files):
            yield self._dataset(crep), self._dataset(rrep)
