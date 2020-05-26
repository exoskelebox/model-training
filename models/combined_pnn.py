from __future__ import absolute_import
import os
from .model import Model
from human_gestures import HumanGestures
import random
from statistics import mean, stdev
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
import time
from utils.data_utils import split, exclude
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import shutil
from ast import literal_eval


class Combined_PNN(Model):
    def __init__(self, name='cpnn', tunable=True, resume=None):
        super().__init__(name=name, tunable=tunable)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        if resume:
            self.logdir = os.path.join(
                'logs', resume)
            self.resumed = True
        else:
            self.logdir = os.path.join(
                'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), self.name]))
            self.resumed = False

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []
        if self.resumed:
            subjects_accuracy = self._resume()

        summary_writer = tf.summary.create_file_writer(
            self.logdir)

        for subject_index, (subject_repetitions, remainder) in enumerate(HumanGestures(batch_size).subject_datasets()):
            if subject_index < 1 + len(subjects_accuracy):
                continue
            subject_logdir = os.path.join(self.logdir, f's{subject_index}')

            subject_summary_writer = tf.summary.create_file_writer(
                subject_logdir)

            k_fold = []
            result = []
            pretrained_weights = []
            tf.print(f'\nSubject {subject_index + 1}')

            for rep_index, (val, train) in enumerate(subject_repetitions, start=1):
                rep_logdir = os.path.join(subject_logdir, f'r{rep_index}')

                tf.print('Repetition {}'.format(rep_index))

                # Firstly train a model on all the data except for the targeted subject
                pretrained_logdir = os.path.join(
                    rep_logdir, 'pretrained')
                pretrained_summary_writer = tf.summary.create_file_writer(
                    pretrained_logdir)

                # Split the dataset according to the given ratio
                pre_train, pre_val = split(remainder, (8, 2))

                pre_train = pre_train.shuffle(2**10)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=pretrained_logdir, profile_batch=0)

                pretrained, model = self.build(hp=kt.HyperParameters())

                if not pretrained_weights:
                    pretrained.fit(
                        pre_train,
                        validation_data=pre_val,
                        epochs=epochs,
                        callbacks=[early_stop, tensorboard])
                    pretrained_weights = pretrained.get_weights()
                else:
                    pretrained.set_weights(pretrained_weights)

                # Freeze the pretrained layers
                for layer in pretrained.layers:
                    layer.trainable = False

                # ... Then we train a model on the targeted subject in context of the other subjects
                val, test = split(val)
                train = train.shuffle(2**10)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=rep_logdir, profile_batch='10,20')

                model.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard]
                )

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = mean(k_fold)
            tf.print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)
            standard_deviation = stdev(k_fold)

            with open(os.path.join(subject_logdir, 'results.txt'), 'w') as f:
                f.write(
                    str({'mean': average, 'stdev': standard_deviation, 'k_fold': k_fold}))

        total_average = mean(subjects_accuracy)
        total_standard_deviation = stdev(subjects_accuracy)
        with open(os.path.join(self.logdir, 'results.txt'), 'w') as f:
            f.write(str({'mean': total_average, 'stdev': total_standard_deviation,
                         'subjects_accuracy': subjects_accuracy}))

        return (total_average, subjects_accuracy)

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

    def _resume(self):
        subjects = [f.path for f in os.scandir(self.logdir) if f.is_dir()]
        subject_accuracies = []
        for subject in subjects:
            try:
                with open(os.path.join(subject, 'results.txt'), 'r') as f:
                    results = literal_eval(f.read())
                    subject_accuracies.append(results['mean'])
            except FileNotFoundError:
                shutil.rmtree(subject)
                continue
        return subject_accuracies
