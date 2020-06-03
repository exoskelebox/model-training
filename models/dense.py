from __future__ import absolute_import
from tensorflow.keras import mixed_precision
import tensorflow as tf
import pandas as pd
import os
from _datetime import datetime
from sklearn.model_selection._split import LeaveOneGroupOut, train_test_split
from kerastuner import HyperModel, HyperParameters
import numpy as np


class Dense(HyperModel):
    def __init__(self, name='dense', tunable=True):
        super().__init__(name=name, tunable=tunable)
        if tf.config.list_physical_devices('GPU'):
            policy = mixed_precision.experimental.Policy('mixed_float16')
            mixed_precision.experimental.set_policy(policy)
        self.built = False

    def run_model(self, batch_size, epochs):
        fname = 'hgest.hdf'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = tf.keras.utils.get_file(
            fname, origin)
        key = 'normalized'
        df = pd.read_hdf(path, key)

        df_test = df[df.repetition == df.repetition.max()]
        df = df[df.repetition != df.repetition.max()]

        subject_ids = df.subject_id.unique()
        sensor_cols = [col for col in df.columns if col.startswith('sensor')]

        subject_results = []

        early_stop = tf.keras.callbacks.EarlyStopping(
            min_delta=0.001, patience=5, restore_best_weights=True)

        logdir = os.path.join(
            'logs', self.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

        for subject_index, subject_id in enumerate(subject_ids, start=1):
            print('Subject {}/{}'.format(subject_index, len(subject_ids)))
            subject_df = df[df.subject_id == subject_id]

            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, str(
                subject_index), 'checkpoint'), save_best_only=True, save_weights_only=True)

            model = self.build(hp=HyperParameters())

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, stratify=y)

            model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                x_val, y_val), callbacks=[early_stop, checkpoint])

            subject_test_df = df_test[df_test.subject_id == subject_id]

            x_test = subject_test_df[sensor_cols].to_numpy()
            y_test = subject_test_df.label.to_numpy()

            evaluation = model.evaluate(x_test, y_test, batch_size)

            subject_results.append(evaluation)

            subject_loss, subject_accuracy = evaluation

            tf.summary.scalar('subject_loss', subject_loss,
                              step=(subject_index - 1))
            tf.summary.scalar('subject_accuracy', subject_accuracy,
                              step=(subject_index - 1))

            model_loss, model_accuracy = np.mean(
                subject_results, axis=0).tolist()

            tf.summary.scalar('mean_loss', model_loss,
                              step=(subject_index - 1))
            tf.summary.scalar('mean_accuracy', model_accuracy,
                              step=(subject_index - 1))

            file_writer.flush()

        mean = np.mean(subject_results, axis=0).tolist()

        file_writer.close()

        return (mean, subject_results)

    def build(self, hp):
        dropout = hp.Float('dropout',
                           min_value=0.2,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2**hp.Int('exponent_1',
                                            min_value=6,
                                            max_value=8,
                                            default=6,
                                            step=1), activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(2**hp.Int('exponent_2',
                                            min_value=6,
                                            max_value=8,
                                            default=6,
                                            step=1), activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(18, activation='softmax', dtype='float32')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        with tf.summary.record_if(not self.built):
            tf.summary.text('hyperparameters', str(hp.values), step=0)
            self.built = True

        return model
