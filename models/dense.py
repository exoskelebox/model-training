from __future__ import absolute_import

from .model import Model
from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
import tensorflow as tf


class Dense(Model):

    def __init__(self, reps, batch_size, epoch):
        self.reps = reps
        self.epoch = epoch
        self.batch_size = batch_size
        self.subject_paths = human_gestures.subject_paths

    def run_model(self):
        subjects_accuracy = []

        for i in range(len(self.subject_paths)):
            k_fold = []
            result = []
            print(f'\nSubject {i + 1}/{len(self.subject_paths)}')

            for n in range(self.reps):
                model = self._model()

                train, val, test = human_gestures.get_data(
                    human_gestures.subject_paths[i], n, self.batch_size)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                model.fit(train,
                          validation_data=val,
                          epochs=self.epoch,
                          callbacks=[early_stop])

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = mean(k_fold)
            print(f'\nmean accuracy: {average}')
            subjects_accuracy.append(average)

        total_average = mean(subjects_accuracy)

        return (total_average, subjects_accuracy)

    def _model(self):
        return tf.keras.Sequential([
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
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(18, activation='softmax')
        ])
