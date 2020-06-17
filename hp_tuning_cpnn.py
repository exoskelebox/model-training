import os
import tensorflow as tf
from models import Dense
import pandas as pd
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization, Sklearn
import kerastuner as kt
from kerastuner import HyperParameters
from datetime import datetime
from sklearn.model_selection._split import train_test_split
from sklearn import metrics, model_selection
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from tensorflow.keras import mixed_precision


def build_model(hp: HyperParameters):
    inputs = tf.keras.Input((15,))
    x = inputs
    y = inputs
    t_dropout = hp.Float('target_dropout', 0.0, 0.5, 0.1, default=0.2)
    p_dropout = hp.Float('pretrain_dropout', 0.0, 0.5, 0.1, default=0.2)

    for i in range(1):
        # hidden layer
        x = tf.keras.layers.Dense(2**hp.Int('target_exponent_{}'.format(i), 5, 8, default=6), activation='relu', kernel_initializer='he_uniform', name='target_dense_{}'.format(i))(x)
        y = tf.keras.layers.Dense(2**hp.Int('pretrain_exponent_{}'.format(i), 5, 8, default=6), activation='relu', kernel_initializer='he_uniform', name='pretrain_dense_{}'.format(i))(y)
        a = tf.keras.layers.Dense(2**hp.Int('adapter_exponent_{}'.format(i), 2, 6, default=4), activation='relu', kernel_initializer='he_uniform', name='target_adapter_{}'.format(i))(y)

        # dropout layer
        x = tf.keras.layers.Dropout(t_dropout, name='target_dropout_{}'.format(i))(x)
        x = tf.keras.layers.concatenate([x, a], name='target_concat_{}'.format(i))
        y = tf.keras.layers.Dropout(p_dropout, name='pretrain_dropout_{}'.format(i))(y)

    x = tf.keras.layers.Dense(18, activation='softmax', dtype='float32', name='target_output')(x)
    y = tf.keras.layers.Dense(18, activation='softmax', dtype='float32', name='pretrain_output')(y)
    model = tf.keras.Model(inputs=inputs, outputs=[x, y])

    return model


class MyTuner(kt.Tuner):
    def run_trial(self, trial, df, batch_size, epochs, callbacks=[]):
        hp = trial.hyperparameters

        subject_results = []

        subject_ids = df.subject_id.unique()
        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        for subject_index, subject_id in enumerate(subject_ids, start=1):
            result = []
            src_df = df[df.subject_id != subject_id]
            subject_df = df[df.subject_id == subject_id]

            repetitions = subject_df.repetition.to_numpy()
            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            src_x = src_df[sensor_cols].to_numpy()
            src_y = src_df.label.to_numpy()

            model = self.hypermodel.build(trial.hyperparameters)

            src_x_train, src_x_val, src_y_train, src_y_val = train_test_split(
                src_x, src_y, stratify=src_y)

            for layer in model.layers:
                if 'target' in layer.name:
                    layer.trainable = False

            model.compile(
                optimizer='adam',
                loss={'pretrain_output': 'sparse_categorical_crossentropy'},
                metrics=['accuracy'])

            early_stop = tf.keras.callbacks.EarlyStopping('val_pretrain_output_accuracy', min_delta=0.001, restore_best_weights=True, patience=10)
            model.fit(src_x_train, src_y_train, batch_size,
                        epochs, validation_data=(src_x_val, src_y_val), callbacks=[early_stop])

            for layer in model.layers:
                if 'target' in layer.name:
                    layer.trainable = True
            model.get_layer('pretrain_output').trainable = False
            weights = model.get_weights()

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=repetitions), start=1):
                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]
                
                model.set_weights(weights)

                model.compile(
                    optimizer='adam',
                    loss={'target_output': 'sparse_categorical_crossentropy'},
                    metrics=['accuracy'])

                early_stop = tf.keras.callbacks.EarlyStopping('val_target_output_accuracy', min_delta=0.001, restore_best_weights=True, patience=10)
                model.fit(x_train, y_train, batch_size,
                          epochs, validation_data=(x_test, y_test), callbacks=[early_stop])

                loss, _, target_output_accuracy, _ = model.evaluate(x_test, y_test, batch_size)
                result.append((loss, target_output_accuracy))

            mean = np.mean(result, axis=0).tolist()

            subject_loss, subject_accuracy = mean

            subject_results.append(mean)

            model_loss, model_accuracy = np.mean(
                subject_results, axis=0).tolist()

        loss, accuracy = np.mean(subject_results, axis=0).tolist()

        self.oracle.update_trial(
            trial.trial_id, {'val_loss': loss, 'val_accuracy': accuracy})


if __name__ == '__main__':
    if tf.config.list_physical_devices('GPU'):
        policy = mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.experimental.set_policy(policy)

    fname = 'hgest.hdf'
    origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
    path: str = tf.keras.utils.get_file(
        fname, origin)
    key = 'normalized'
    df = pd.read_hdf(path, key)
    df = df[df.repetition != df.repetition.max()]

    tuner = MyTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('val_accuracy', 'max'),
            max_trials=10),
        hypermodel=build_model,
        directory='hp',
        project_name='test')  # datetime.now().strftime("%Y%m%d-%H%M%S"))

    tuner.search_space_summary()

    tuner.search(
        df,
        2**9,
        100,
        [tf.keras.callbacks.EarlyStopping('val_accuracy', min_delta=0.001, restore_best_weights=True, patience=10)])
    tuner.results_summary(5)
    # print(tuner.get_best_hyperparameters(5))
