import tensorflow as tf
from models.pnn import PNN_Column, PNN_Model
from datasets import normalized_human_gestures as human_gestures
import time
import os

feature_layer = human_gestures.get_feature_layer([
    'subject_gender',
    'subject_age',
    'subject_fitness',
    'subject_handedness',
    'subject_wrist_circumference',
    'subject_forearm_circumference',
    #'repetition',
    'readings',
    'wrist_calibration_iterations',
    'wrist_calibration_values',
    'arm_calibration_iterations',
    'arm_calibration_values'
])

simple_feature_layer = human_gestures.get_feature_layer([
    'readings'
])

"""
adapter = [tf.keras.layers.Dense(16, 'relu')]
core = [
    tf.keras.layers.Dense(32, 'relu'), 
    tf.keras.layers.Dense(32, 'relu'),
    tf.keras.layers.Dense(18, 'softmax') ]
"""

adapters = {'type': tf.keras.layers.Dense, 'units': 16, 'activation': 'relu'}
core = [
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 18, 'activation': 'softmax'}]
layer_info = {'core': core, 'adapters': adapters}

adapters2 = {'type': tf.keras.layers.Dense, 'units': 16, 'activation': 'relu'}
core2 = [
    {'type': tf.keras.layers.Dense, 'units': 32, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 32, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 18, 'activation': 'softmax'}]
layer_info2 = {'core': core, 'adapters': adapters}


subject_paths = human_gestures.subject_paths
num_subjects = len(subject_paths)
num_repetitions = 5
results = {i:{rep:{} for rep in range(num_repetitions)} for i in range(num_subjects)}
sequence = []

def train_and_eval(model, train, test):
    print('Model constructed. Compiling...')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print('Model compiled.')
    print('Creating callbacks...')
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0.0001,
        patience=5,
        restore_best_weights=True)

    print('Callbacks created.')
    print('Fitting model...')
    model.fit(train,
            validation_data=test,
            epochs=100,
            callbacks=[earlystop_callback])

    print('Model fitted.')
    print('Evaluating model...')
    result = model.evaluate(test)
    print(result)
    return result

model_desc = f"dense_32_32_18"
t = time.strftime('%m-%d-%H%M%S', time.localtime(time.time()))
file_name = f"{model_desc} - {t}.txt"
path = os.path.join('results', file_name)

with open(path,"w+") as f:
    subjects = enumerate(subject_paths)
    # for each repetition
    for rep in range(num_repetitions):
        for i in range(num_subjects):
            
            print(f'Loading data for repetition {rep}, subject {i}:')
            train_cur, test_cur = human_gestures.get_data(subject_paths[i], rep, batch_size=1024)
            #print('Loading data for other subjects:')
            #train_old, test_old = human_gestures.get_data_except(subject_paths[i], rep, batch_size=1024)
            print(f'Data loading complete.')

            column = PNN_Column(layer_info, generation=0)
            columns = [column]

            model = PNN_Model(feature_layer=simple_feature_layer, columns=columns)

            results[i][rep]['column'] = train_and_eval(model, train_cur, test_cur)
            model.summary()


            seq = tf.keras.Sequential([
                feature_layer,
                tf.keras.layers.Dense(32, activation='relu'),
                #tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                #tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(18, activation='softmax')
            ])
            results[i][rep]['sequential'] = train_and_eval(seq, train_cur, test_cur)
            seq.summary()
            print('Models Evaluated.')

            diff = results[i][rep]['column'][1] - results[i][rep]['sequential'][1]
            results[i][rep]['diff'] = diff


            f.write(f"Subject {i}, rep {rep}: {results[i][rep]}/n")