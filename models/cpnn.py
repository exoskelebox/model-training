from __future__ import absolute_import
from tensorflow.keras import callbacks, layers, mixed_precision, models, utils
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection._split import LeaveOneGroupOut, train_test_split
import numpy as np
from kerastuner import HyperParameters, HyperModel
from _datetime import datetime


class CombinedProgressiveNeuralNetwork(HyperModel):
    def __init__(self, name='cpnn', tunable=True):
        super().__init__(name=name, tunable=tunable)
        if tf.config.list_physical_devices('GPU'):
            policy = mixed_precision.experimental.Policy('mixed_float16')
            mixed_precision.experimental.set_policy(policy)
        self.built = False

    def run_model(self, batch_size, epochs):
        fname = 'hgest.hdf'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = utils.get_file(
            fname, origin)
        key = 'normalized'
        df = pd.read_hdf(path, key)

        subject_results = []

        src_early_stop = callbacks.EarlyStopping(
            restore_best_weights=True, patience=2)
        tar_early_stop = callbacks.EarlyStopping(
            'val_accuracy', restore_best_weights=True, patience=10)

        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        logdir = os.path.join(
            'logs', self.name, datetime.now().strftime("%Y%m%d-%H%M%S"))

        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

        subject_ids = df.subject_id.unique()

        for subject_index, subject_id in enumerate(subject_ids, start=1):
            tf.print('Subject {}/{}'.format(subject_index, len(subject_ids)))
            result = []

            src_df = df[df.subject_id != subject_id]

            x = src_df[sensor_cols].to_numpy()
            y = src_df.label.to_numpy()

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, stratify=y)

            src_model, _ = self.build(hp=HyperParameters())

            src_model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                x_test, y_test), callbacks=[src_early_stop])

            src_weights = src_model.get_weights()

            tar_df = df[df.subject_id == subject_id]
            repetitions = tar_df.repetition.to_numpy()
            x = tar_df[sensor_cols].to_numpy()
            y = tar_df.label.to_numpy()

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=repetitions), start=1):

                # Recreating the model is VERY important, as the adam optimizer needs to be reset.
                src_model, tar_model = self.build(hp=HyperParameters())
                src_model.set_weights(src_weights)

                for layer in src_model.layers:
                    layer.trainable = False

                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]

                checkpoint = callbacks.ModelCheckpoint(os.path.join(logdir, str(
                    subject_index), str(rep_index), 'checkpoint'), save_best_only=True, save_weights_only=True)

                tar_model.fit(x_train, y_train, batch_size, epochs, validation_data=(
                    x_test, y_test), callbacks=[tar_early_stop, checkpoint])

                result.append(tar_model.evaluate(
                    x_test, y_test, batch_size))

            mean = np.mean(result, axis=0).tolist()

            subject_loss, subject_accuracy = mean

            tf.summary.scalar('subject_loss', subject_loss,
                              step=(subject_index - 1))
            tf.summary.scalar('subject_accuracy', subject_accuracy,
                              step=(subject_index - 1))

            subject_results.append(mean)

            model_loss, model_accuracy = np.mean(
                subject_results, axis=0).tolist()

            tf.summary.scalar('mean_loss', model_loss,
                              step=(subject_index - 1))
            tf.summary.scalar('mean_accuracy', model_accuracy,
                              step=(subject_index - 1))

            file_writer.flush()

        mean = np.mean(subject_results, axis=0).tolist()

        file_writer.close()

        return (mean, subject_results)

    def build(self, hp):
        exponent = hp.Int('exponent',
                          min_value=4,
                          max_value=10,
                          default=8,
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

        inputs = layers.Input((15,))

        # 1st hidden layer
        x = layers.Dense(2**exponent, activation='relu')(inputs)
        y = layers.Dense(2**exponent, activation='relu')(inputs)
        a = layers.Dense(2**adapter_exponent, activation='relu')(y)

        # 1st dropout layer
        x = layers.Dropout(dropout)(x)
        x = layers.concatenate([x, a])
        y = layers.Dropout(dropout)(y)

        """ # 2nd hidden layer
        x = layers.Dense(2**exponent, activation='relu')(x)
        y = layers.Dense(2**exponent, activation='relu')(y)
        a = layers.Dense(2**adapter_exponent, activation='relu')(y)

        # 2nd dropout layer
        x = layers.Dropout(dropout)(x)
        x = layers.concatenate([x, a])
        y = layers.Dropout(dropout)(y) """

        # Output layer
        y = layers.Dense(18, activation='softmax',
                         dtype='float32')(y)

        pretrained_model = models.Model(
            inputs=inputs, outputs=y)

        pretrained_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        x = layers.Dense(18, activation='softmax', dtype='float32')(x)

        model = models.Model(
            inputs=inputs, outputs=x)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        with tf.summary.record_if(not self.built):
            tf.summary.text('hyperparameters', str(hp.values), step=0)
            self.built = True

        return pretrained_model, model
