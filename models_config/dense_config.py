# from datasets import human_gestures
from datasets import normalized_human_gestures as human_gestures
from models_config import base_config
import tensorflow as tf
import statistics
import random

feature_layer = human_gestures.get_feature_layer([
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
])


class old_dense_model(base_config.base):

    def __init__(self, reps, batch_size, epoch):
        self.reps = reps
        self.epoch = epoch
        self.batch_size = batch_size
        self.subject_paths = human_gestures.subject_paths
        self.model = self._old_dense_model()

    def run_model(self):
        subjects_accuracy = []
        random.shuffle(self.subject_paths)

        for i in range(len(self.subject_paths)):
            k_fold = []
            result = []

            for n in range(self.reps):
                train, test = human_gestures.get_data(
                    human_gestures.subject_paths[i], n, self.batch_size)

                early_stop_callback = self._early_stop()
                self._compile_model(self.model)

                self.model.fit(train,
                               validation_data=test,
                               epochs=self.epoch,
                               callbacks=[early_stop_callback])

                result = self.model.evaluate(test)
                k_fold.append(result[-1])

            average = statistics.mean(k_fold)
            subjects_accuracy.append(average)

        total_average = statistics.mean(subjects_accuracy)
        print(f"model's average for all participants: {total_average}")

        return (total_average, subjects_accuracy)

    def _old_dense_model(self):
        return tf.keras.Sequential([
            feature_layer,
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(18, activation='softmax')
        ])
