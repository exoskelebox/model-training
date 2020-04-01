from datasets import human_gestures
from utils.data_utils import fraction_train_test_split, feature_train_test_split
import tensorflow as tf
import statistics
from models import pnn
layers = [{'type':tf.keras.layers.Dense, 'units':2, 'activation':'relu'} for i in range(3)]
train, test = human_gestures.get_data(human_gestures.subject_paths[0], 3, batch_size=1024)

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

print(results)

print("Avg. ACC = {}".format(statistics.mean(
    [acc for key, (loss, acc) in results.items()]))) """
layers = [
    {'type':tf.keras.layers.Dense, 'units':64, 'activation':'relu'},
    {'type':tf.keras.layers.Dense, 'units':64, 'activation':'relu'},
    {'type':tf.keras.layers.Dense, 'units':18, 'activation':'softmax'}]



model = pnn.PNN_Model(layers, feature_layer=feature_layer)

print('Model constructed. Compiling...')
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Model compiled.')
print('Creating callbacks...')
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001,
    patience=10)

print('Callbacks created.')
print('Fitting model...')
model.fit(train,
          validation_data=test,
          epochs=20,
          callbacks=[earlystop_callback])

print('Model fitted.')
print('Evaluating model...')
result = model.evaluate(test)
print(result)
print('Model Evaluated.')
previous = [model]
for i in range(3):
    model2 = pnn.PNN_Model(layers, feature_layer=feature_layer, previous=previous)

    print('Model constructed. Compiling...')
    model2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    print('Model compiled.')
    print('Creating callbacks...')
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001,
        patience=10)

    print('Callbacks created.')
    print('Fitting model...')
    model2.fit(train,
            validation_data=test,
            epochs=20,
            callbacks=[earlystop_callback])

    print('Model fitted.')
    print('Evaluating model...')
    result = model2.evaluate(test)
    print(result)
    print('Model Evaluated.')
    previous.append(model)

