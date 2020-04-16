import tensorflow as tf
from models.pnn2 import PNN_Column, PNN_Model
from datasets import normalized_human_gestures as human_gestures

feature_layer = human_gestures.get_feature_layer([
    'readings',
    'subject_gender'
])

adapter = [tf.keras.layers.Dense(16, 'relu')]
core = [tf.keras.layers.Dense(64, 'relu'), tf.keras.layers.Dense(64, 'relu')]

subject_paths = human_gestures.subject_paths
columns = []
results = {}
sequence = []

for i in range(len(subject_paths)):
    sequence.append(i)
    train, test = human_gestures.get_data(
        subject_paths[i], 1, batch_size=1024)

    column = PNN_Column(adapter, core, generation=i)
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
