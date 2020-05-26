from __future__ import absolute_import
import os
from .model import Model
from human_gestures import HumanGestures
import random
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from utils.iter_utils import fold
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
import numpy as np


class PNN(Model):
    def __init__(self, name=datetime.now().strftime("%Y%m%d-%H%M%S"), tunable=True):
        super().__init__(name=name, tunable=tunable)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    def run_model(self, batch_size, epochs):
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

        subject_ids = df.subject_id.unique()
        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        logdir = os.path.join('logs', self.name)

        for subject_index, (target_subject_id, source_subject_ids) in enumerate(fold(subject_ids), start=1):
            tf.print('Subject {}/{}'.format(subject_index, len(subject_ids)))
            result = []

            columns = self.build(hp=kt.HyperParameters(),
                                 num_columns=len(subject_ids))

            for column_index, subject_id in enumerate(source_subject_ids):
                tf.print('Column {}/{}'.format(column_index +
                                               1, len(source_subject_ids)))
                subject_df = df[df.subject_id == subject_id]

                x = subject_df[sensor_cols].to_numpy()
                y = subject_df.label.to_numpy()

                x_train, x_val, y_train, y_val = train_test_split(
                    x, y, stratify=y)

                model: tf.keras.Model = columns[column_index]

                model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                    x_val, y_val), callbacks=[early_stop])

                for layer in model.layers[1:]:
                    layer.trainable = False

            model = columns[-1]
            base_weights = model.get_weights()

            subject_df = df[df.subject_id == target_subject_id]

            repetitions = subject_df.repetition.to_numpy()
            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=repetitions), start=1):
                model.set_weights(base_weights)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(logdir, str(subject_index), str(rep_index)), profile_batch=0)

                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]

                x_val, x_test, y_val, y_test = train_test_split(
                    x_test, y_test, test_size=0.5, stratify=y_test)

                model.fit(x_train, y_train, batch_size,
                          epochs, validation_data=(x_val, y_val), callbacks=[early_stop, tensorboard])

                result.append(model.evaluate(x_test, y_test, batch_size))

            mean = np.mean(result, axis=0).tolist()
            subject_results.append(mean)

        mean = np.mean(subject_results, axis=0).tolist()
        return (mean, subject_results)

    def build(self, hp=kt.HyperParameters(), num_columns=20):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
                          step=1)
        adapter_exponent = hp.Int('adapter_exponent',
                                  min_value=2,
                                  max_value=6,
                                  default=4,
                                  step=1)
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        columns = []

        # Input layer
        inputs = tf.keras.layers.Input((15,))

        for i in tqdm.trange(num_columns, desc='Building columns'):

            # Hidden 1
            x = tf.keras.layers.Dense(
                2**exponent, activation='relu', name='dense_1_{}'.format(i))(inputs)
            x = tf.keras.layers.Dropout(0.2, name='dropout_1_{}'.format(i))(x)

            ada_x = [tf.keras.layers.Dense(2**adapter_exponent, activation='relu', name='adapter_1_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_1_{}'.format(j)).output) for j in range(i)]

            x = tf.keras.layers.concatenate(
                [x, *ada_x], name='concat_1_{}'.format(i)) if ada_x else x

            # Hidden 2
            x = tf.keras.layers.Dense(
                2**exponent, activation='relu', name='dense_2_{}'.format(i))(x)
            x = tf.keras.layers.Dropout(0.2, name='dropout_2_{}'.format(i))(x)

            ada_x = [tf.keras.layers.Dense(2**adapter_exponent, activation='relu', name='adapter_2_{}_{}'.format(
                i, j))(columns[j].get_layer('dense_2_{}'.format(j)).output) for j in range(i)]

            x = tf.keras.layers.concatenate(
                [x, *ada_x], name='concat_2_{}'.format(i)) if ada_x else x

            # Output
            outputs = tf.keras.layers.Dense(
                18, activation='softmax', name='output_{}'.format(i), dtype='float32')(x)

            model = tf.keras.models.Model(
                inputs=inputs, outputs=outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Choice('learning_rate',
                              values=[1e-2, 1e-3, 1e-4], default=1e-3)),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

            columns.append(model)

        return columns
