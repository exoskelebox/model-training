import tensorflow as tf
from models.pnn import PNN_Column, PNN_Model
from datasets import normalized_human_gestures as human_gestures
import time
import os

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

pretrained_core = [
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 64, 'activation': 'relu'},
    {'type': tf.keras.layers.Dense, 'units': 18, 'activation': 'softmax'}]
pretrained_layer_info = {'core': pretrained_core, 'adapters': adapters}


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
        patience=10,
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

model_desc = f"pnn_C_2x64-2x32_A_16-mixed_features"
t = time.strftime('%m-%d-%H%M%S', time.localtime(time.time()))
file_name = f"{model_desc} - {t}.txt"
path = os.path.join('results', file_name)

with open(path,"w+") as f:
    f.write(f"Model: Combined PNN - Pretrained: {pretrained_core} - Current: {core} - Adapters: {adapters}\n")
    f.write(f"Features: Pretrained: {features} - Current: {simple_features} \n")


subjects = enumerate(subject_paths)
# for each repetition
for rep in range(num_repetitions):
    for i in range(num_subjects):
         
        print(f'Loading data for repetition {rep}, subject {i}:')
        train_cur, test_cur = human_gestures.get_data(subject_paths[i], rep, batch_size=1024)
        print('Loading data for other subjects:')
        train_old, test_old = human_gestures.get_data_except(subject_paths[i], rep, batch_size=1024)
        print(f'Data loading complete.')

        old_column = PNN_Column(layer_info, generation=0, feature_layer=feature_layer)
        columns = [old_column]
        old_model = PNN_Model(columns=columns)
        results[i][rep]['pretrained'] = train_and_eval(old_model, train_old, test_old)

        column = PNN_Column(layer_info, generation=1, feature_layer=simple_feature_layer)
        columns = [old_column, column]
        model = PNN_Model(columns=columns)
        results[i][rep]['current'] = train_and_eval(model, train_cur, test_cur)

        old_model.summary()            
        model.summary()

        print('Models Evaluated.')

        #diff = results[i][rep]['column'][1] - results[i][rep]['sequential'][1]
        #results[i][rep]['diff'] = diff

        with open(path, "a+") as f:
            f.write(f"Subject {i}, rep {rep}: {results[i][rep]}\n")