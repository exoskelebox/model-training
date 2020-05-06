from __future__ import absolute_import
import os
from .model import Model
from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
import tensorflow as tf
import kerastuner as kt


class Dense(Model):

    def __init__(self):
        self.subject_paths = human_gestures.subject_paths

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, subject_path in enumerate(self.subject_paths):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, _ in enumerate(next(os.walk(subject_path))[-1]):

                train, val, test = human_gestures.get_data(
                    subject_path, rep_index, batch_size)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                model = self.build()

                model.fit(train,
                          validation_data=val,
                          epochs=epochs,
                          callbacks=[early_stop])
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
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)

        model = tf.keras.Sequential([
            human_gestures.get_feature_layer([
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
