import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.pnn import PNN_Column, PNN_Model
from datasets import normalized_human_gestures as human_gestures
import time
import os
import statistics

features = [
    #'subject_gender',
    #'subject_age',
    #'subject_fitness',
    #'subject_handedness',
    #'subject_wrist_circumference',
    #'subject_forearm_circumference',
    #'repetition',
    'readings',
    #'wrist_calibration_iterations',
    #'wrist_calibration_values',
    #'arm_calibration_iterations',
    #'arm_calibration_values'
]
feature_layer = human_gestures.get_feature_layer(features)

simple_features = ['readings']
simple_feature_layer = human_gestures.get_feature_layer(simple_features)


feature_column_inputs = {
    'subject_id': layers.Input(name='subject_id', shape=(), dtype=tf.int64),
    'subject_gender': layers.Input(name='subject_gender', shape=(), dtype=tf.string),
    'subject_age': layers.Input(name='subject_age', shape=(), dtype=tf.int64),
    'subject_fitness':layers.Input(name='subject_fitness', shape=(), dtype=tf.int64),
    'subject_handedness': layers.Input(name='subject_handedness', shape=(), dtype=tf.string),
    'subject_impairment': layers.Input(name='subject_impairment', shape=(), dtype=tf.string),
    'subject_wrist_circumference': layers.Input(name='subject_wrist_circumference', shape=(), dtype=tf.float32),
    'subject_forearm_circumference': layers.Input(name='subject_forearm_circumference', shape=(), dtype=tf.float32),
    'repetition': layers.Input(name='repetition', shape=(), dtype=tf.int64),
    'reading_count': layers.Input(name='reading_count', shape=(), dtype=tf.int64),
    'readings': layers.Input(name='readings', shape=(15,)),
    'arm_calibration_iterations': layers.Input(name='arm_calibration_iterations', shape=(), dtype=tf.uint16),
    'arm_calibration_values': layers.Input(name='arm_calibration_values', dtype=tf.dtypes.int64, shape=(8,)),
    'wrist_calibration_iterations': layers.Input(name='wrist_calibration_iterations', shape=(), dtype=tf.int64),
    'wrist_calibration_values': layers.Input(name='wrist_calibration_values', dtype=tf.int64, shape=(7,)),
    'timedelta': layers.Input(name='timedelta', shape=(), dtype=tf.int64),
}


subject_paths = human_gestures.subject_paths
num_subjects = len(subject_paths)
num_repetitions = 5
results = {i:{rep:{} for rep in range(num_repetitions)} for i in range(num_subjects)}
sequence = []


model_desc = f"pretrained_test"
t = time.strftime('%m-%d-%H%M%S', time.localtime(time.time()))
file_name = f"{model_desc} - {t}.txt"
path = os.path.join('results', file_name)

def log(s):
    with open(path,'a+') as f:
        print(s, file=f)

log(f"Model: Pretrained \n")
log(f"Subjects: All \n")
log(f"Features: {features} \n")
arcitecture_logged=False

def dense_layers(inputs):
    # Hidden layers
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x) 
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(18, activation='softmax')(x)
    return outputs

def train_and_eval(model, train, val, test):
    print('Compiling model...')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print('Model compiled.')
    print('Creating callbacks...')
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0.0001,
        patience=10,
        restore_best_weights=True)

    print('Callbacks created.')
    print('Fitting model...')
    model.fit(train,
            validation_data=val,
            epochs=200,
            callbacks=[earlystop_callback])

    print('Model fitted.')
    print('Evaluating model...')
    result = model.evaluate(test)
    print(result)
    return result[-1]

acc = []
# 5-fold
for i in range(1):
    for k in range(5):
        print(f'Loading data for fold {k}:')
        #data = (data.map(lambda x,y: (simple_feature_layer(x), y)) for data in human_gestures.get_data_except(batch_size=1024, ratios=(8,1,1), offset = 2*k))
        data = human_gestures.get_data_except(batch_size=1024, ratios=(8,1,1), offset = 2*k)
        train, val, test = data
        print(f'Data loading complete.')

        model_input = layers.Input(name='readings', batch_size=1024, shape=(15,), dtype='float32')
        #x = simple_feature_layer(model_input)
        model_output = dense_layers(model_input)
        model = keras.models.Model(inputs=model_input, outputs=model_output)


        

        if not arcitecture_logged:
            model.summary(print_fn=log)
            arcitecture_logged=True

        result = train_and_eval(model, train, val, test)
        acc.append(result)
        log(f"Fold {k} accuracy: {result}")

mean = statistics.mean(acc)
stdev = statistics.stdev(acc)
log(f"Mean: {mean} \nSt.Dev.: {stdev}")

