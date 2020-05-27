import kerastuner as kt
import os
import pandas as pd
import tensorflow as tf


class KFoldTuner(kt.Tuner):
    def __init__(self, oracle, hypermodel, max_model_size=None, optimizer=None, loss=None, metrics=None, distribution_strategy=None, directory=None, project_name=None, logger=None, tuner_id=None, overwrite=False):
        super().__init__(oracle, hypermodel, max_model_size=max_model_size, optimizer=optimizer, loss=loss, metrics=metrics,
                         distribution_strategy=distribution_strategy, directory=directory, project_name=project_name, logger=logger, tuner_id=tuner_id, overwrite=overwrite)
        fname = 'hgest.hdf'
        origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
        path: str = tf.keras.utils.get_file(
            fname, origin)
        key = 'normalized'
        self.df = pd.read_hdf(path, key)

    def run_trial(self, trial, epochs=1, batch_size=32):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.0001, restore_best_weights=True,
            patience=10)

        trial_val_accuracies = []
        hp = trial.hyperparameters
        subject_paths = human_gestures.subject_paths
        for subject_index, subject_path in enumerate(subject_paths):
            reps = next(os.walk(subject_path))[-1]
            model_val_accuracies = []
            for rep_index, _ in enumerate(reps):
                print(f'Subject: {subject_index + 1}/{len(subject_paths)}')
                print(f'Rep: {rep_index + 1}/{len(reps)}')

                train, val, test = human_gestures.get_data(
                    subject_path, rep_index, batch_size)
                model = self.hypermodel.build(hp)
                model.fit(train, validation_data=val, epochs=epochs)
                model_val_accuracies.append(model.evaluate(test)[-1])
                print()

            trial_val_accuracies.append(mean(model_val_accuracies))

        self.oracle.update_trial(
            trial.trial_id, {'val_accuracy': mean(trial_val_accuracies)})
