from __future__ import absolute_import

""" from . import utils
from . import datasets
from . import test
from . import database """
import tensorflow as tf


file = tf.keras.utils.get_file(
    'hgest', 'https://storage.googleapis.com/exoskelebox/hgest.tfrecord')
print(file)
