from __future__ import absolute_import
import os
from .model import Model
from datasets import normalized_human_gestures as human_gestures
import random
from statistics import mean
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
from callbacks import ConfusionMatrix


class FineTuned(Model):
    def __init__(self):
        self.subject_paths = human_gestures.subject_paths

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, subject_path in enumerate(self.subject_paths):
            k_fold = []
            result = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, _ in enumerate(next(os.walk(subject_path))[-1]):

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'finetuning', f's{subject_index}', f'r{rep_index}']))
                pre_logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'pre_finetuning', f's{subject_index}', f'r{rep_index}']))
                
                train, val, test = human_gestures.get_data(
                    subject_path, rep_index, batch_size)
                
                pre_train, pre_val, pre_test = human_gestures.get_data_except(
                    [subject_path], batch_size, offset=2*rep_index)
                
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                pretrained = self.build()

                pre_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=pre_logdir)
                pre_cm = ConfusionMatrix(pre_test, pretrained, pre_logdir)

                pretrained.fit(
                    pre_train,
                    validation_data=pre_val,
                    epochs=epochs,
                    callbacks=[early_stop, pre_tensorboard, pre_cm]
                )

                # Freeze the pretrained layers
                for layer in pretrained.layers:
                    layer.trainable = False

                model = self._switch_head(pretrained)

                # Check the trainable status of the individual layers
                for layer in model.layers:
                    print(layer, layer.trainable)

                tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
                cm = ConfusionMatrix(test, model, logdir)

                # Warm up the new head
                model.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard, cm]
                )

                # Unfreeze the pretrained layers
                for layer in pretrained.layers:
                    layer.trainable = True

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate',
                                values=[1e-3, 1e-4, 1e-5], default=1e-4)),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

                # Finetune the model
                model.fit(
                    train,
                    validation_data=val,
                    epochs=epochs,
                    callbacks=[early_stop, tensorboard, cm]
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
        finetuning_exponent = hp.Int('finetuning_exponent',
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
            tf.keras.layers.Dense(2**finetuning_exponent, activation='relu'),
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

    def _switch_head(self, model, hp=kt.HyperParameters()):
        finetuning_exponent = hp.Int('finetuning_exponent',
                          min_value=4,
                          max_value=10,
                          default=6,
                          step=1)
                           
        dropout = hp.Float('dropout',
                           min_value=0.0,
                           default=0.2,
                           max_value=0.5,
                           step=0.1)
                           
        new_head = [
            tf.keras.layers.Dense(2**finetuning_exponent, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(18, activation='softmax')
        ]

        model = self._rec_switch(model, new_head)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4], default=1e-3)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def _rec_switch(self, model, new:list, hp=kt.HyperParameters()):
        if len(new) >= 1:
            model.pop()
            model = self._rec_switch(model, new[1:])
            model.add(new[0])
            return model
        else:
            return model
