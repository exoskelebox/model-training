import tensorflow as tf
import types
from pathlib import Path
from utils.iter_utils import fold
import os
from itertools import tee
from datetime import datetime


class HumanGestures():
    def __init__(self, batch_size=64):
        self._batch_size = batch_size

    @staticmethod
    def path() -> Path:
        fname = 'hgest_half.tar.gz'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = tf.keras.utils.get_file(
            fname, origin, extract=True, file_hash='4179d27b9771e683878db9f927464391')
        return Path(path.rsplit('.', 2)[0])

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
                example_protos, {
                    # 'subject_id': tf.io.FixedLenFeature([], tf.int64),
                    # 'gesture': tf.io.FixedLenFeature([], tf.string),
                    # 'repetition': tf.io.FixedLenFeature([], tf.int64),
                    'readings': tf.io.FixedLenFeature(shape=(15), dtype=tf.float32),
                    'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
                })
            return parsed_examples['readings'], parsed_examples['label']

        if isinstance(file_or_files, (list, types.GeneratorType)):
            files = list(map(str, file_or_files))

            dataset = tf.data.Dataset\
                .from_tensor_slices(files)\
                .shuffle(buffer_size=len(files), reshuffle_each_iteration=False)\
                .interleave(
                    map_func=tf.data.TFRecordDataset,
                    cycle_length=tf.data.experimental.AUTOTUNE,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.TFRecordDataset(str(file_or_files))

        return dataset\
            .batch(
                batch_size=self.batch_size, drop_remainder=True)\
            .map(
                map_func=parse_batch,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )\
            .cache()

    def dataset(self):
        """
        Returns a dataset with all the human gestures data
        """
        return self._dataset(self.path().rglob('*.tfrecord'))

    def subject_datasets(self, fold_current=True, flatten_remainder=True):
        subject_paths = [Path(f.path)
                         for f in os.scandir(self.path()) if f.is_dir()]

        for csub, rsub in fold(subject_paths):
            csub_reps = list(csub.glob('*.tfrecord'))
            if flatten_remainder:
                rsub_reps = self._dataset(
                    [rep for sub in rsub for rep in sub.glob('*.tfrecord')])
            else:
                rsub_reps = [self._repetition_datasets(
                    [rep for rep in sub.glob('*.tfrecord')]) for sub in rsub]
            yield (self._repetition_datasets(csub_reps) if fold_current else self._dataset(csub_reps)), rsub_reps

    def _repetition_datasets(self, files):
        for crep, rrep in fold(files):
            yield self._dataset(crep), self._dataset(rrep)
