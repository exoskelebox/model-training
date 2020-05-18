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


class Dense(Model):

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, (subject_repetitions, _) in enumerate(HumanGestures(batch_size).subject_datasets()):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, (val, train) in enumerate(subject_repetitions):
                val, test = split(val)
                train = train.shuffle(2**14)

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'dense', f's{subject_index}', f'r{rep_index}']))

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

                model = self.build(hp=kt.HyperParameters())

                cm = ConfusionMatrix(test, model, logdir)

                model.fit(train,
                          validation_data=val,
                          epochs=epochs,
                          callbacks=[early_stop, tensorboard, cm])

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = mean(k_fold)
            print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)

        total_average = mean(subjects_accuracy)

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
            HumanGestures.feature_layer([
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
            ]),
            tf.keras.layers.Dense(2**exponent, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(2**exponent, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(18, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model
