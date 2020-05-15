import tensorflow as tf
import time
from tqdm import tqdm
import pathlib
import os
import types


class HumanGesturesDataset(tf.data.Dataset):

    @staticmethod
    def feature_description():
        """
        Returns a feature description dictionary.
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
        Returns a feature input dictionary. Used for the construction of models using the functional Keras API.
        """
        return {
            'subject_id': tf.keras.Input((1,), dtype=tf.int64, name='subject_id'),
            'gesture': tf.keras.Input((1,), dtype=tf.string, name='gesture'),
            'repetition': tf.keras.Input((1,), dtype=tf.int64, name='repetition'),
            'readings': tf.keras.Input((15,), dtype=tf.float32, name='readings')
        }

    @staticmethod
    def feature_layer(cols: [] = None) -> tf.keras.layers.Dense:
        feature_description = HumanGesturesDataset.feature_description()

        valid_cols = feature_description.keys()

        # Label is not a valid column for the feature layer
        valid_cols.remove('label')

        # Ensure unique columns
        cols = set(cols)

        if not cols:
            cols = valid_cols

        if not set(cols).issubset(valid_cols):
            exit(
                f"One or more invalid feature column names, valid feature column names are {', '.join(valid_cols)}.")

        return tf.keras.layers.DenseFeatures([feature_description[col](col) for col in cols])

    @staticmethod
    def path():
        """
        Returns the path to the dataset directory.
        """
        fname = 'hgest.tar.gz'
        origin = 'https://storage.googleapis.com/exoskelebox/hgest.tar.gz'
        path: str = tf.keras.utils.get_file(
            fname, origin, extract=True, file_hash='452ec3c4e89eb187825725092f9c6a8e')
        return path.rsplit('.', 2)[0]

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
                batch_size=self._batch_size,
                drop_remainder=True)\
            .shuffle(2 ** 14, reshuffle_each_iteration=False)\
            .map(
                map_func=parse_batch,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\


    def subject_shards(self):
        paths = [f.path for f in os.scandir(self.path()) if f.is_dir()]

        for path in paths:
            files = path.glob('*tfrecord')
            yield self._dataset(files)

    def __new__(cls, key='normalized', batch_size=64):
        cls._batch_size = batch_size
        path = pathlib.Path(os.path.join(cls.path(), key))
        return cls._dataset(cls, path.glob('*.tfrecord'))


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


dataset = HumanGesturesDataset(batch_size=1024)
print(dataset.feature_description())
