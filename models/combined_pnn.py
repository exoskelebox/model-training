from __future__ import absolute_import
import os
from .model import Model
from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
import time


class Combined_PNN(Model):
    def __init__(self):
        self.subject_paths = human_gestures.subject_paths
        self.feature_layer = human_gestures.get_feature_layer([
            # 'subject_gender',
            # 'subject_age',
            # 'subject_fitness',
            # 'subject_handedness',
            # 'subject_wrist_circumference',
            # 'subject_forearm_circumference',
            # 'repetition',
            'readings',
            # 'wrist_calibration_iterations',
            # 'wrist_calibration_values',
            # 'arm_calibration_iterations',
            # 'arm_calibration_values'
        ])

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, subject_path in enumerate(self.subject_paths):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, _ in enumerate(next(os.walk(subject_path))[-1]):

                # Firstly train a model on all the data except for the targeted subject
                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'pre_cpnn', f's{subject_index}', f'r{rep_index}']))

                data = human_gestures.get_data_except(
                    [subject_path], batch_size, offset=2*rep_index)

                train, val, test = (
                    d.map(lambda x, y: (self.feature_layer(x), y)) for d in data)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=logdir)

                pretrained, model = self.build()

                pretrained.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard]
                )

                # Freeze the pretrained layers
                for layer in pretrained.layers:
                    layer.trainable = False

                # ... Then we train a model on the targeted subject in context of the other subjects
                data = human_gestures.get_data(
                    subject_path, rep_index, batch_size)

                train, val, test = (
                    d.map(lambda x, y: (self.feature_layer(x), y)) for d in data)

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'cpnn', f's{subject_index}', f'r{rep_index}']))

                tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

                model.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard]
                )

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = mean(k_fold)
            print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)

        total_average = mean(subjects_accuracy)

        return (total_average, subjects_accuracy)

    def build(self, hp=kt.HyperParameters()):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
                          step=1)
        adapter_exponent = hp.Int('exponent',
                                  min_value=2,
                                  max_value=6,
                                  default=4,
                                  step=1)
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        # Input layer
        model_input = layers.Input(
            name='readings', shape=(15,), dtype='float32')

        # 1st hidden layer
        pretrained_dense_1 = layers.Dense(2**exponent, activation='relu')
        y = pretrained_dense_1(model_input)
        adapter_1 = layers.Dense(2**adapter_exponent, activation='relu')
        a = adapter_1(y)
        dense_1 = layers.Dense(2**exponent, activation='relu')
        x = dense_1(model_input)

        # 1st dropout layer
        pretrained_dropout_1 = layers.Dropout(dropout)
        y = pretrained_dropout_1(y)
        dropout_1 = layers.Dropout(dropout)
        x = dropout_1(x)

        # 2nd hidden layer
        x = layers.concatenate([x, a])
        pretrained_dense_2 = layers.Dense(2**exponent, activation='relu')
        y = pretrained_dense_2(y)
        adapter_2 = layers.Dense(2**adapter_exponent, activation='relu')
        a = adapter_2(y)
        dense_2 = layers.Dense(2**exponent, activation='relu')
        x = dense_2(x)

        # 2nd dropout layer
        pretrained_dropout_2 = layers.Dropout(dropout)
        y = pretrained_dropout_2(y)
        dropout_2 = layers.Dropout(dropout)
        x = dropout_2(x)

        # 3rd hidden layer
        x = layers.concatenate([x, a])
        pretrained_dense_3 = layers.Dense(2**exponent, activation='relu')
        y = pretrained_dense_3(y)
        adapter_3 = layers.Dense(2**adapter_exponent, activation='relu')
        a = adapter_3(y)
        dense_3 = layers.Dense(2**exponent, activation='relu')
        x = dense_3(x)

        # 3rd dropout layer
        pretrained_dropout_3 = layers.Dropout(dropout)
        y = pretrained_dropout_3(y)
        dropout_3 = layers.Dropout(dropout)
        x = dropout_3(x)

        # Output layer
        pretrained_output = layers.Dense(18, activation='softmax')(y)
        pretrained_model = keras.models.Model(
            inputs=model_input, outputs=pretrained_output)
        pretrained_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        x = layers.concatenate([x, a])
        model_output = layers.Dense(18, activation='softmax')(x)
        model = keras.models.Model(inputs=model_input, outputs=model_output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return pretrained_model, model
