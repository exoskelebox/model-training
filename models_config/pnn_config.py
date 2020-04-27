import tensorflow as tf
from models.pnn import PNN_Column, PNN_Model
from models_config import base_config
from datasets import normalized_human_gestures as human_gestures
import random
import statistics

feature_layer = human_gestures.get_feature_layer([
    'subject_gender',
    'subject_age',
    'subject_fitness',
    'subject_handedness',
    'subject_wrist_circumference',
    'subject_forearm_circumference',
    'repetition',
    'readings',
    'wrist_calibration_iterations',
    'wrist_calibration_values',
    'arm_calibration_iterations',
    'arm_calibration_values'
])


class PNN(base_config.base):

    def __init__(self, reps, batch_size, epoch):
        self.reps = reps
        self.epoch = epoch
        self.batch_size = batch_size
        self.columns = []
        self.subject_paths = human_gestures.subject_paths
        self.layer_info = self._pnn_config()

    def run_model(self):
        subjects_accuracy = []
        random.shuffle(self.subject_paths)

        for i in range(len(self.subject_paths)):
            k_fold = []
            result = []

            self._create_column(i, self.layer_info)
            model = self._pnn_model(i)

            for n in range(self.reps):
                train, test = human_gestures.get_data(
                    human_gestures.subject_paths[i], n, self.batch_size)

                early_stop_callback = self._early_stop()
                self._compile_model(model)

                model.fit(train,
                          validation_data=test,
                          epochs=self.epoch,
                          callbacks=[early_stop_callback])

                result = model.evaluate(test)
                k_fold.append(result[-1])

            average = statistics.mean(k_fold)
            subjects_accuracy.append(average)

        total_average = statistics.mean(subjects_accuracy)
        print(f"model's average for all participants: {total_average}")

        return (total_average, subjects_accuracy)

    def _pnn_model(self, generation_index):
        return PNN_Model(feature_layer=feature_layer, columns=self.columns)

    def _create_column(self, generation_index, layer_info):
        column = PNN_Column(layer_info, generation=generation_index)
        self.columns.append(column)

    def _pnn_config(self):
        adapters = {'type': tf.keras.layers.Dense,
                    'units': 16, 'activation': 'relu'}

        core = [
            {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
            {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
            {'type': tf.keras.layers.Dense, 'units': 18, 'activation': 'softmax'}]

        return {'core': core, 'adapters': adapters}
