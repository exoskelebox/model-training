import kerastuner as kt
import os
from datasets import normalized_human_gestures as human_gestures
from statistics import mean


class KFoldTuner(kt.Tuner):

    def run_trial(self, trial, epochs=1, batch_size=32):
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
