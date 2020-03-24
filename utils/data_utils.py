from __future__ import absolute_import
import tensorflow as tf


def fraction_train_test_split(dataset: tf.data.Dataset, split=0.8, shuffle=True, random_state=None) -> (tf.data.Dataset, tf.data.Dataset):
    if split >= 1.0 or split <= 0:
        exit(f"Invalid split value, valid range is between 0.0 and 1.0.")

    num_samples = len(list(dataset))
    train_samples = int(split * num_samples)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples, seed=random_state)

    return dataset.take(train_samples), dataset.skip(train_samples)

def feature_train_test_split(dataset: tf.data.Dataset, split_feature:str, is_test_func=None, shuffle=True, random_state=None) -> (tf.data.Dataset, tf.data.Dataset):
    if not split_feature in dataset.element_spec[0]:
        exit(f"Invalid split feature. not present in dataset.")

    @tf.function
    def is_test(x, y):
        return is_test_func(x[split_feature])
    
    @tf.function
    def is_training(x, y):
        return not is_test(x, y)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(list(dataset)), seed=random_state)

    return dataset.filter(is_training), dataset.filter(is_test)
