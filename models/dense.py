from __future__ import absolute_import
import os
from .model import Model
import random
from statistics import mean
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix
from human_gestures import HumanGestures
from utils.data_utils import feature_fold, shuffle, split
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Dense(Model):
    def __init__(self, name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, (subject_repetitions, _) in enumerate(HumanGestures(batch_size).subject_datasets()):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, (val, train) in enumerate(subject_repetitions):
                val, test = split(val)
                train = train.shuffle(2**10)

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'dense', f's{subject_index}', f'r{rep_index}']))

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)
                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=logdir, profile_batch='10,20')

                model = self.build(hp=kt.HyperParameters())

                model.fit(train,
                          validation_data=val,
                          epochs=epochs,
                          callbacks=[tensorboard])

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = mean(k_fold)
            print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)

            subject_average = tf.summary.create_file_writer(
                os.path.join(logdir, 'model_average'))
            with subject_average.as_default():
                tf.summary.text(f"subject_{subject_index}_average", str(
                    subjects_accuracy), step=0)

        total_average = mean(subjects_accuracy)

        model_average = tf.summary.create_file_writer(
            os.path.join(logdir, 'model_average'))
        with model_average.as_default():
            tf.summary.text(f"model_average", str(
                (total_average, subjects_accuracy)), step=1)

        return (total_average, subjects_accuracy)

    def build(self, hp):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
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
