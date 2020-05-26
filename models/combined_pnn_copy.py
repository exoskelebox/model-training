from __future__ import absolute_import
import os
from .model import Model
from human_gestures import HumanGestures
import random
from statistics import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
import time
from utils.data_utils import split, exclude
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
import numpy as np


class Combined_PNN(Model):
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

        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        logdir = os.path.join('logs', self.name)

        subject_ids = df.subject_id.to_numpy()
        repetitions = df.repetition.to_numpy()
        x = df[sensor_cols].to_numpy()
        y = df.label.to_numpy()

        logo = LeaveOneGroupOut().split(x, y, groups=subject_ids)
        [logo.next() for _ in range(skip)]

        for subject_index, (source_index, target_index) in enumerate(logo, start=(skip + 1)):
            print('Subject {}/{}'.format(subject_index, len(np.unique(subject_ids))))
            x_source, y_source = x[source_index], y[source_index]

            x_train, x_test, y_train, y_test = train_test_split(
                x_source, y_source, stratify=y_source)

            result = []

            source_model, target_model = self.build(hp=kt.HyperParameters())

            source_model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                x_test, y_test), callbacks=[early_stop])

            target_weights = target_model.get_weights()

            x_target, y_target = x[target_index], y[target_index]
            repetitions_target = repetitions[target_index]

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x_target, y_target, groups=repetitions_target)):
                target_model.set_weights(target_weights)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(logdir, str(subject_index), str(rep_index)), profile_batch=0)

                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]

                target_model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                    x_test, y_test), callbacks=[early_stop, tensorboard])

                result.append(target_model.evaluate(
                    x_test, y_test, batch_size))

            mean = np.mean(result, axis=0).tolist()

            subject_results.append(mean)

        mean = np.mean(subject_results, axis=0).tolist()

        return (mean, subject_results)

    def build(self, hp=kt.HyperParameters()):
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

        inputs = tf.keras.layers.Input((15,))

        # 1st hidden layer
        x = layers.Dense(2**exponent, activation='relu')(inputs)
        y = layers.Dense(2**exponent, activation='relu')(inputs)
        a = layers.Dense(2**adapter_exponent, activation='relu')(y)

        # 1st dropout layer
        x = layers.Dropout(dropout)(x)
        x = layers.concatenate([x, a])
        y = layers.Dropout(dropout)(y)

        # 2nd hidden layer
        x = layers.Dense(2**exponent, activation='relu')(x)
        y = layers.Dense(2**exponent, activation='relu')(y)
        a = layers.Dense(2**adapter_exponent, activation='relu')(y)

        # 2nd dropout layer
        x = layers.Dropout(dropout)(x)
        x = layers.concatenate([x, a])
        y = layers.Dropout(dropout)(y)

        # Output layer
        y = layers.Dense(18, activation='softmax',
                         dtype='float32')(y)

        pretrained_model = keras.models.Model(
            inputs=inputs, outputs=y)

        pretrained_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-2),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        x = layers.Dense(18, activation='softmax', dtype='float32')(x)

        model = keras.models.Model(
            inputs=inputs, outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-2),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return pretrained_model, model
