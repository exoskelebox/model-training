import logging
import os
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from models.pnn import PNN_Column, PNN_Model
from models_config.dense_config import old_dense_model
from models_config.pnn_config import pnn_config, pnn_model, create_column
from utils.data_utils import fraction_train_test_split, feature_train_test_split
from datasets import normalized_human_gestures as human_gestures
import statistics
import random

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



def run_models(selected_models = [], reps=5):
    subjects_accuracy = []
    columns = []
    
    models_config = {
        'old_dense' : lambda: old_dense_model(),
        'pnn'   : lambda: pnn_model(*args)
    }
    
    subject_paths = human_gestures.subject_paths
    random.shuffle(subject_paths)

    for current_model in selected_models:
        for i in range(len(subject_paths)):
            k_fold = []
            print(f"\ntraining on participant: {i}")
            try:
                columns.append(create_column(generation_index = i, layer_info=pnn_config()))
                args = (i, columns)
                
                model = models_config[current_model]()
                
                for n in range(reps):
                    train, test = human_gestures.get_data(
                        human_gestures.subject_paths[i], n, 1024)

                    early_stop_callback = early_stop()
                    compile_model(model)

                    model.fit(train,
                                validation_data=test,
                                epochs=5,
                                callbacks=[early_stop_callback])

                    result = model.evaluate(test)
                    print(f"\nModel Evaluated.")

                    k_fold.append(result[-1])

                average = statistics.mean(k_fold)
                subjects_accuracy.append(average)

            except KeyError:
                print('Error: Invalid model "%s"' % current_model)
                continue

        total_average = statistics.mean(subjects_accuracy)
        print(f"model's average for all participants: {total_average}")



def compile_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print('Model compiled.')


def early_stop():
    print('Creating callbacks...')
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001,
        patience=3)


if __name__ == "__main__":
    run_models(['pnn','old_dense'])
    
