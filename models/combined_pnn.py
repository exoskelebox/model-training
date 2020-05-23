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


class Combined_PNN(Model):
    def __init__(self, name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    def run_model(self, batch_size, epochs):
        subjects_accuracy = []

        for subject_index, (subject_repetitions, remainder) in enumerate(HumanGestures(batch_size).subject_datasets()):
            k_fold = []
            result = []
            pretrained_weights = []
            print(f'\nSubject {subject_index + 1}')

            for rep_index, (val, train) in enumerate(subject_repetitions):
                # Firstly train a model on all the data except for the targeted subject
                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'pre_cpnn', f's{subject_index}', f'r{rep_index}']))

                # Split the dataset according to the given ratio
                pre_train, pre_val = split(remainder, (8, 2))

                pre_train = pre_train.shuffle(2**10)

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', min_delta=0.0001, restore_best_weights=True,
                    patience=10)

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=logdir, profile_batch='10,20')

                pretrained, model = self.build(hp=kt.HyperParameters())

                if not pretrained_weights:
                    pretrained.fit(
                        pre_train,
                        validation_data=pre_val,
                        epochs=epochs,
                        callbacks=[early_stop, tensorboard]
                    )
                    pretrained_weights = pretrained.get_weights()
                else:
                    pretrained.set_weights(pretrained_weights)

                # Freeze the pretrained layers
                for layer in pretrained.layers:
                    layer.trainable = False

                # ... Then we train a model on the targeted subject in context of the other subjects
                val, test = split(val)
                train = train.shuffle(2**10)

                logdir = os.path.join(
                    'logs', '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), 'cpnn', f's{subject_index}', f'r{rep_index}']))

                tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=logdir, profile_batch='10,20')

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
            # tf.summary.text("Confusion Matrix", cm_image, step=epoch)

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
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return pretrained_model, model
