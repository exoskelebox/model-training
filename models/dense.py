# from datasets import human_gestures
from datasets import normalized_human_gestures as human_gestures
import tensorflow as tf

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


def old_dense_model():
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(18, activation='softmax')
    ])
    print('Model constructed. Compiling...')
    return model


if __name__ == "__main__":
    old_dense_model()
    # train, test = human_gestures.get_data(human_gestures.subject_paths[0], 0, 1024)
    # i = 0
    # for feature_batch, label_batch in train:
    # print('Every feature:', list(feature_batch.keys()))
    # print('A batch of calibration:', feature_batch['wrist_calibration_values'])
    # print('A batch of calibration:', feature_batch['arm_calibration_values'])
    # print('A batch of targets:', label_batch )
