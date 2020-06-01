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

        subject_ids = df.subject_id.unique()
        sensor_cols = [col for col in df.columns if col.startswith('sensor')]

        # Split the dataset into a train and test dataset
        test_df = df[df.repetition == df.repetition.max()]
        train_df = df.drop(test_df.index)

        del df

        subject_results = []

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', restore_best_weights=True, patience=10)

        logdir = os.path.join(
            'logs', self.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

        for subject_index, subject_id in enumerate(subject_ids, start=1):
            print('Subject {}/{}'.format(subject_index, len(subject_ids)))
            test_subject_df = test_df[test_df.subject_id == subject_id]
            train_subject_df = train_df[train_df.subject_id == subject_id]

            result = []

            train_repetitions = train_subject_df.repetition.to_numpy()
            x = train_subject_df[sensor_cols].to_numpy()
            y = train_subject_df.label.to_numpy()

            for rep_index, (train_index, val_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=train_repetitions), start=1):

                x_train, y_train = x[train_index], y[train_index]
                x_val, y_val = x[val_index], y[val_index]

                checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, str(
                    subject_index), str(rep_index), 'checkpoint'), save_best_only=True, save_weights_only=True)

                model = self.build(hp=HyperParameters())
                model.fit(x_train, y_train, batch_size,
                          epochs, validation_data=(x_val, y_val), callbacks=[early_stop, checkpoint])

                result.append(model.evaluate(
                    test_subject_df[sensor_cols].to_numpy(),
                    test_subject_df.label.to_numpy(),
                    batch_size))

            mean = np.mean(result, axis=0).tolist()

            subject_loss, subject_accuracy = mean

            tf.summary.scalar('subject_loss', subject_loss,
                              step=(subject_index - 1))
            tf.summary.scalar('subject_accuracy', subject_accuracy,
                              step=(subject_index - 1))

            subject_results.append(mean)

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
                           default=0.5,
                           max_value=0.5,
                           step=0.1)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2**hp.Int('exponent_1',
                                            min_value=6,
                                            max_value=8,
                                            default=7,
                                            step=1), activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(18, activation='softmax', dtype='float32')
        ])

        """
        tf.keras.layers.Dense(2**hp.Int('exponent_2',
                                            min_value=6,
                                            max_value=8,
                                            default=8,
                                            step=1), activation='relu'),
            tf.keras.layers.Dropout(dropout), """

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        with tf.summary.record_if(not self.built):
            tf.summary.text('hyperparameters', str(hp.values), step=0)
            self.built = True

        return model
