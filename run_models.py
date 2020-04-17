import tensorflow as tf
from models.pnn2 import PNN_Column, PNN_Model
from models.dense import old_dense_model
from datasets import normalized_human_gestures as human_gestures
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
    subjects_average = []
    model = old_dense_model()

    for i in range(len(human_gestures.subject_paths)):
        k_fold = []

        for n in range(5):
            train, test = human_gestures.get_data(
                human_gestures.subject_paths[i], n, 1024)

            early_stop_callback = early_stop()
            compile_model(model)

            model.fit(train,
                      validation_data=test,
                      epochs=5,
                      callbacks=[early_stop_callback])

            result = model.evaluate(test)
            print(f"Model Evaluated.")

            k_fold.append(result[-1])

        average = statistics.mean(k_fold)
        subjects_average.append(average)

        print(f"Average for participant {i} is {average} ")

    total = statistics.mean(subjects_average)
    print(f"model's average for all participants: {total}")
