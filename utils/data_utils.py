from __future__ import absolute_import
import tensorflow as tf


def train_test_split(dataset: tf.data.Dataset, split=0.8, shuffle=True, random_state=None) -> (tf.data.Dataset, tf.data.Dataset):
    if split >= 1.0 or split <= 0:
        exit(f"Invalid split value, valid range is between 0.0 and 1.0.")

    num_samples = len(list(dataset))
    train_samples = int(split * num_samples)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples, seed=random_state)

    return dataset.take(train_samples), dataset.skip(train_samples)
