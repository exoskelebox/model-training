import tensorflow as tf
from tensorflow import feature_column
from database import Database
import typing
from typing import List, Tuple
from collections import defaultdict

# A utility method to identify test data


def is_test(x, y) -> bool:
    return x['reading_count'] % 5 == 0

# A utility method to identify training data


def is_training(x, y) -> bool:
    return not is_test(x, y)


def split(dataset, split) -> (tf.data.Dataset, tf.data.Dataset):
    def recover(x, y): return y
    def is_a(x, y): return x % split != 0
    def is_b(x, y): return not is_a(x, y)
    a_dataset = dataset.enumerate() \
        .filter(is_a) \
        .map(recover)
    b_dataset = dataset.enumerate() \
        .filter(is_b) \
        .map(recover)
    return a_dataset, b_dataset


# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column, example_batch) -> None:
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    features, labels = example_batch
    print(feature_column.name)
    print(feature_layer(features).numpy())


feature_column_constructors = {
    # TODO: Decide if this should be included
    'subject_id': lambda x=None: get_numeric_column('subject_id', tf.uint16, x),
    'subject_gender': lambda x=None: get_indicator_column('subject_gender', ['m', 'f'], x),
    'subject_age': lambda x=None: get_bucketized_column('subject_age', [18, 25, 30, 35, 40, 45, 50, 55, 60, 65], tf.uint8, x),
    'subject_fitness': lambda x=None: get_bucketized_column('subject_fitness', [2, 4, 6, 8], tf.uint8, x),
    'subject_handedness': lambda x=None: get_indicator_column('subject_handedness', ['l', 'r', 'a'], x),
    'subject_impairment': lambda x=None: get_indicator_column('subject_impairment', ['True', 'False'], x),
    'subject_wrist_circumference': lambda x=None: get_numeric_column('subject_wrist_circumference', tf.float32, x),
    'subject_forearm_circumference': lambda x=None: get_numeric_column('subject_forearm_circumference', tf.float32, x),
    'gesture': None,  # TODO: Handle Label
    'repetition': None,
    'reading_count': lambda x=None: get_numeric_column('reading_count', tf.uint16, x),
    'timestamp': None,  # TODO: Handle timestamps
    'readings': lambda x=None: get_numeric_array_column('readings', 15, tf.uint8, x),
    'arm_calibration_gesture': None,  # TODO: Decide if this should be included
    'arm_calibration_iterations': lambda x=None: get_numeric_column('arm_calibration_iterations', tf.uint16, x),
    'arm_calibration_values': lambda x=None: get_numeric_array_column('arm_calibration_values', 8, tf.uint8, x),
    'wrist_calibration_gesture': None,  # TODO: Decide if this should be included
    'wrist_calibration_iterations': lambda x=None: get_numeric_column('wrist_calibration_iterations', tf.uint16, x),
    'wrist_calibration_values': lambda x=None: get_numeric_array_column('wrist_calibration_values', 7, tf.uint8, x),
}


def get_feature_columns_test(example_batch) -> List['feature_column']:

    columns = []
    for featurename in example_batch[0].keys():
        constructor = feature_column_constructors[featurename]
        if constructor is not None:
            columns.append(constructor(example_batch))
    return columns


def get_feature_columns(featurenames) -> List['feature_column']:
    return [feature_column_constructors[featurename]() for featurename in featurenames]


def get_categorical_column_with_identity(name, example_batch=None):
    column = feature_column.categorical_column_with_identity(name, 2)
    column_one_hot = feature_column.indicator_column(column)
    if example_batch:
        demo(column_one_hot, example_batch)
    return column_one_hot


def get_numeric_column(name, dtype=tf.float32, example_batch=None) -> feature_column.numeric_column:
    column = feature_column.numeric_column(name, dtype=dtype)
    if example_batch:
        demo(column, example_batch)
    return column


def get_numeric_array_column(name, shape, dtype=tf.float32, example_batch=None) -> feature_column.numeric_column:
    column = feature_column.numeric_column(name, shape=shape, dtype=dtype)
    if example_batch:
        demo(column, example_batch)
    return column


def get_bucketized_column(name, bucket_boundaries, dtype=tf.float32, example_batch=None) -> feature_column.bucketized_column:
    column = feature_column.numeric_column(name, dtype=dtype)
    bucketized_column = feature_column.bucketized_column(
        column, boundaries=bucket_boundaries)
    if example_batch:
        demo(bucketized_column, example_batch)
    return bucketized_column


def get_indicator_column(name, vocabulary, example_batch=None) -> feature_column.indicator_column:
    column = feature_column.categorical_column_with_vocabulary_list(
        name, vocabulary)
    column_one_hot = feature_column.indicator_column(column)
    if example_batch:
        demo(column_one_hot, example_batch)
    return column_one_hot


LABEL = 'gesture'
FEATURES = [
    'subject_id',
    'subject_gender',
    'subject_age',
    'subject_fitness',
    'subject_handedness',
    'subject_impairment',
    'subject_wrist_circumference',
    'subject_forearm_circumference',
    # 'gesture',
    # 'repetition',
    'reading_count',
    # 'timestamp',
    'readings',
    # 'wrist_calibration_gesture',
    'wrist_calibration_iterations',
    'wrist_calibration_values',
    # 'arm_calibration_gesture',
    'arm_calibration_iterations',
    'arm_calibration_values',
]


def get_data(features=FEATURES, label=LABEL) -> Tuple[tf.data.Dataset]:
    """
    Retreives data from database.
    Returns (train, test) tf.data.Datasets.
    """

    # Convert to set to ensure uniqueness
    colnames = set(features + [label])
    cur = Database().query(
        f"SELECT {','.join(colnames)} FROM training LIMIT 5")
    data_batched = defaultdict(list)
    [[data_batched[feature].append(
        row[index]) for index, feature in enumerate(colnames)] for row in cur]
    data_batched = dict(data_batched)
    labels = data_batched.pop(label)  # {'label': data_batched.pop(label)}

    # Convert timestamps to iso format
    if 'timestamp' in data_batched.keys():
        print(data_batched['timestamp'][:5])
        data_batched['timestamp'] = list(
            map(lambda x: x.isoformat(), data_batched['timestamp']))
        print(data_batched['timestamp'][:5])

    # Convert boolean values to strings
    if 'subject_impairment' in data_batched:
        print(data_batched['subject_impairment'][:5])
        data_batched['subject_impairment'] = list(
            map(str, data_batched['subject_impairment']))
        print(data_batched['subject_impairment'][:5])

    tf_data = tf.data.Dataset.from_tensor_slices((data_batched, labels))
    print(len(list(tf_data)))

    print('Filtering:')
    train = tf_data.filter(is_training)
    test = tf_data.filter(is_test)

    print(f'{len(list(train))}/{len(list(test))}')

    return train, test


def run_model(train, test) -> None:
    batch_size = 1  # A small batch sized is used for demonstration purposes
    train = train.shuffle(buffer_size=len(list(train))).batch(batch_size)

    test = test.shuffle(buffer_size=len(list(test)))
    test, val = split(test, 5)
    test = test.batch(batch_size)
    val = val.batch(batch_size)

    # We will use this batch to demonstrate feature columns
    example_batch = next(iter(train))
    print(example_batch)
    exit()

    get_feature_columns_test(example_batch)

    feature_columns = get_feature_columns(example_batch[0].keys())
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    print('Feature layer constructed.')
    print('Constructing model...')
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
    print(result)
    print('Model Evaluated.')


def main():
    train, test = get_data()
    run_model(train, test)


if __name__ == "__main__":
    main()
