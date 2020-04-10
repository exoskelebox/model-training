from datasets import human_gestures
from utils.data_utils import fraction_train_test_split, feature_train_test_split
import tensorflow as tf
import statistics
from models.pnn import PNN_Column, PNN_Model
import os
#train, test = human_gestures.get_data(human_gestures.subject_paths[0], 3, batch_size=1024)

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

results = {}
#dataset = dataset.batch(1)

""" for i in range(5):
    train, test = feature_train_test_split(
        dataset, split_feature='repetition', is_test_func=lambda x: x == i)
    test, val = fraction_train_test_split(test)
    train, test, val = train.batch(1), test.batch(1), val.batch(1)

    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    print('Model constructed. Compiling...')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Model compiled.')
    print('Creating callbacks...')

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001,
        patience=3)

    print('Callbacks created.')
    print('Fitting model...')
    model.fit(train,
              validation_data=val,
              epochs=20,
              callbacks=[earlystop_callback])

    print('Model fitted.')
    print('Evaluating model...')
    result = model.evaluate(test)
    results[i] = result
    print(result)
    print('Model Evaluated.')
-
print(results)

print("Avg. ACC = {}".format(statistics.mean(
    [acc for key, (loss, acc) in results.items()]))) """


adapters = {'type':tf.keras.layers.Dense, 'units':16, 'activation':'relu'} 
core = [
    {'type':tf.keras.layers.Dense, 'units':64, 'activation':'relu'},
    {'type':tf.keras.layers.Dense, 'units':64, 'activation':'relu'},
    {'type':tf.keras.layers.Dense, 'units':18, 'activation':'softmax'}]
layer_info = {'core':core, 'adapters':adapters}

columns = []
results = {}
sequence = []
for i in range(len(human_gestures.subject_paths)):
    sequence.append(i)
    train, test = human_gestures.get_data(human_gestures.subject_paths[i], 1, batch_size=1024)

    column = PNN_Column(layer_info, generation=i)
    columns.append(column)
    model = PNN_Model(feature_layer=feature_layer, columns=columns)

    print('Model constructed. Compiling...')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print('Model compiled.')
    print('Creating callbacks...')
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001,
        patience=3)

    print('Callbacks created.')
    print('Fitting model...')
    model.fit(train,
            validation_data=test,
            epochs=5,
            callbacks=[earlystop_callback])

    print('Model fitted.')
    print('Evaluating model...')
    result = model.evaluate(test)
    print(result)
    results[i] = result
    print('Model Evaluated.')
    model.summary()
    # Save the weights
    save_path = f"pnn/subject_{'_'.join([str(n) for n in sequence])}"
    #save_dir = os.path.dirname(save_path)
    column.save_weights(save_path)
