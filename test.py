from datasets import human_gestures
from utils.data_utils import fraction_train_test_split, feature_train_test_split
import tensorflow as tf
import statistics
dataset = human_gestures.get_data([
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
], 'gesture')

results = {}
#dataset = dataset.batch(1)

for i in range(5):
    train, test = feature_train_test_split(dataset, split_feature='repetition', is_test_func=lambda x: x == i)
    test, val = fraction_train_test_split(test)
    train, test, val = train.batch(1), test.batch(1), val.batch(1)
    model = tf.keras.Sequential([
        human_gestures.get_feature_layer(dataset),
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

print("Avg. ACC = {}".format(statistics.mean([acc for key, (loss, acc) in results.items()])))