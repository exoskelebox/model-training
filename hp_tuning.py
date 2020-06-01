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


def build_model(hp: HyperParameters):
    inputs = tf.keras.Input((15,))
    x = inputs
    dropout = hp.Float('dropout', 0.0, 0.5, 0.1, default=0.2)
    for i in range(2):
        x = tf.keras.layers.Dense(
            2**hp.Int('exponent_{}'.format(i), 5, 8, default=6), 'relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(18, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        'adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


class MyTuner(kt.Tuner):

    def run_trial(self, trial, df, batch_size, epochs, callbacks=[]):
        hp = trial.hyperparameters

        subject_results = []

        subject_ids = df.subject_id.unique()
        sensor_cols = [
            col for col in df.columns if col.startswith('sensor')]

        for subject_index, subject_id in enumerate(subject_ids, start=1):
            subject_df = df[df.subject_id == subject_id]

            result = []

            repetitions = subject_df.repetition.to_numpy()
            x = subject_df[sensor_cols].to_numpy()
            y = subject_df.label.to_numpy()

            for rep_index, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(x, y, groups=repetitions), start=1):

                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]

                model = self.hypermodel.build(trial.hyperparameters)
                model.fit(x_train, y_train, batch_size,
                          epochs, validation_data=(x_test, y_test), callbacks=callbacks)

                result.append(model.evaluate(x_test, y_test, batch_size))

            mean = np.mean(result, axis=0).tolist()

            subject_loss, subject_accuracy = mean

            subject_results.append(mean)

            model_loss, model_accuracy = np.mean(
                subject_results, axis=0).tolist()

        loss, accuracy = np.mean(subject_results, axis=0).tolist()

        self.oracle.update_trial(
            trial.trial_id, {'val_loss': loss, 'val_accuracy': accuracy})


if __name__ == '__main__':
    fname = 'hgest.hdf'
    origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
    path: str = tf.keras.utils.get_file(
        fname, origin)
    key = 'normalized'
    df = pd.read_hdf(path, key)

    tuner = MyTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('val_accuracy', 'max'),
            max_trials=10),
        hypermodel=build_model,
        directory='hp',
        project_name='2d-dense')  # datetime.now().strftime("%Y%m%d-%H%M%S"))

    tuner.search_space_summary()

    tuner.search(
        df,
        2**9,
        1000,
        [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    tuner.results_summary(5)
    #print(tuner.get_best_hyperparameters(5))