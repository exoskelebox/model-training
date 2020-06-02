from __future__ import absolute_import
from tensorflow.keras import callbacks, mixed_precision, models, optimizers, utils
import tensorflow as tf
import pandas as pd
import os
from utils.iter_utils import fold
from kerastuner import HyperParameters, HyperModel
import numpy as np
from sklearn.model_selection._split import LeaveOneGroupOut, train_test_split
import tqdm
from tensorflow.python.keras import layers
from _datetime import datetime


class ProgressiveNeuralNetwork(HyperModel):
    def __init__(self, name='pnn', tunable=True):
        super().__init__(name=name, tunable=tunable)
        if tf.config.list_physical_devices('GPU'):
            policy = mixed_precision.experimental.Policy('mixed_float16')
            mixed_precision.experimental.set_policy(policy)
        self.built = False

    def run_model(self, batch_size, epochs):
        fname = 'hgest.hdf'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = utils.get_file(
            fname, origin)
        key = 'normalized'
        df = pd.read_hdf(path, key)

        subject_results = []

        early_stop = tf.keras.callbacks.EarlyStopping(
            min_delta=0.001, patience=5, restore_best_weights=True)

        subject_ids = df.subject_id.unique()
        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        col_weights = []

        logdir = os.path.join(
            'logs', self.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

        for subject_index, (target_subject_id, source_subject_ids) in enumerate(fold(subject_ids), start=1):
            tf.print('Subject {}/{}'.format(subject_index, len(subject_ids)))

            columns = self.build(hp=HyperParameters(),
                                 num_columns=len(subject_ids))

            for column_index, subject_id in enumerate(source_subject_ids):
                tf.print('Column {}/{}'.format(column_index +
                                               1, len(source_subject_ids)))

                model: models.Model = columns[column_index]

                if column_index < subject_index - 2:
                    model.set_weights(col_weights[column_index])
                else:
                    subject_df = df[df.subject_id == subject_id]
                    x = subject_df[sensor_cols].to_numpy()
                    y = subject_df.label.to_numpy()

                    x_train, x_val, y_train, y_val = train_test_split(
                        x, y, stratify=y)

                    model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                        x_val, y_val), callbacks=[early_stop])

                for layer in model.layers:
                    layer.trainable = False

            model = columns[-1]

            subject_df = df[df.subject_id == target_subject_id]
            subject_test_df = subject_df[subject_df.repetition ==
                                         subject_df.repetition.max()]
            subject_df = subject_df[subject_df.repetition !=
                                    subject_df.repetition.max()]

            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, str(
                subject_index), 'checkpoint'), save_best_only=True, save_weights_only=True)

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, stratify=y)

            model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                x_val, y_val), callbacks=[early_stop, checkpoint])

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

            col_weights = [col.get_weights() for col in columns]

        mean = np.mean(subject_results, axis=0).tolist()

        file_writer.close()

        return (mean, subject_results)

    def build(self, hp, num_columns=20):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=7,
                          step=1)
        adapter_exponent = 4
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.5,
                           max_value=0.5,
                           step=0.1)

        columns = []

        # Input layer
        inputs = layers.Input((15,))

        for i in tqdm.trange(num_columns, desc='Building columns'):

            # Hidden 1
            x = layers.Dense(
                2**exponent, activation='relu', name='dense_1_{}'.format(i))(inputs)
            x = layers.Dropout(0.2, name='dropout_1_{}'.format(i))(x)

            ada_x = [layers.Dense(2**adapter_exponent, activation='relu', name='adapter_1_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_1_{}'.format(j)).output) for j in range(i)]

            x = layers.concatenate(
                [x, *ada_x], name='concat_1_{}'.format(i)) if ada_x else x

            """ # Hidden 2
            x = layers.Dense(
                2**exponent, activation='relu', name='dense_2_{}'.format(i))(x)
            x = layers.Dropout(0.2, name='dropout_2_{}'.format(i))(x)

            ada_x = [layers.Dense(2**adapter_exponent, activation='relu', name='adapter_2_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_2_{}'.format(j)).output) for j in range(i)]

            x = layers.concatenate(
                [x, *ada_x], name='concat_2_{}'.format(i)) if ada_x else x
            """
            # Output
            outputs = layers.Dense(
                18, activation='softmax', name='output_{}'.format(i), dtype='float32')(x)

            model = models.Model(
                inputs=inputs, outputs=outputs)

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            columns.append(model)

        with tf.summary.record_if(not self.built):
            tf.summary.text('hyperparameters', str(hp.values), step=0)
            self.built = True

        return columns
