from __future__ import absolute_import
import os
from .model import Model
import random
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import pandas as pd
from utils.iter_utils import fold
from sklearn.model_selection import LeaveOneGroupOut


class Dense(Model):
    def __init__(self, name=datetime.now().strftime("%Y%m%d-%H%M%S"), tunable=True):
        super().__init__(name=name, tunable=tunable)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    def run_model(self, batch_size, epochs, skip=0):
        fname = 'hgest.hdf'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = tf.keras.utils.get_file(
            fname, origin)
        key = 'normalized'
        df = pd.read_hdf(path, key)

        subject_results = []

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.0001, restore_best_weights=True,
            patience=10)

        subject_ids = df.subject_id.unique()[skip:]
        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        logdir = os.path.join('logs', self.name)

        for subject_index, subject_id in enumerate(subject_ids, start=(skip + 1)):
            print('Subject {}/{}'.format(subject_index, len(subject_ids)))
            subject_df = df[df.subject_id == subject_id]

            result = []

            repetitions = subject_df.repetition.to_numpy()
            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=repetitions), start=1):

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(logdir, str(subject_index), str(rep_index)), profile_batch=0)

                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]
                model = self.build(hp=kt.HyperParameters())
                model.fit(x_train, y_train, batch_size,
                          epochs, validation_data=(x_test, y_test), callbacks=[early_stop, tensorboard])

                result.append(model.evaluate(x_test, y_test, batch_size))

            mean = np.mean(result, axis=0).tolist()

            subject_results.append(mean)

        mean = np.mean(subject_results, axis=0).tolist()

        return (mean, subject_results)

    def build(self, hp):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=7,
                          step=1)
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2**exponent, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(2**exponent, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(18, activation='softmax', dtype='float32')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model
