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

class old_dense_model:

    def __init__(self):
        pass

    def _old_dense_model(self):
        self.model = tf.keras.Sequential([
            feature_layer,
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(18, activation='softmax')
        ])
        print('Model constructed. Compiling...')
    
    def _compile_model(self, model):
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        print('Model compiled.')

    def _early_stop(self):
        print('Creating callbacks...')
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.0001,
            patience=3)

