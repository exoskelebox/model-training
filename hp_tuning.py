import os
import tensorflow as tf
from models import Dense
import pandas as pd
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from datetime import datetime
from sklearn.model_selection._split import train_test_split


fname = 'hgest.hdf'
origin = f'https://storage.googleapis.com/exoskelebox/{fname}'
path: str = tf.keras.utils.get_file(
    fname, origin)
key = 'normalized'
df = pd.read_hdf(path, key)

max_rep = df.repetition.max()
sensor_cols = [col for col in df.columns if col.startswith('sensor')]

train_df = df[df.repetition != max_rep]
val_df = df[df.repetition == max_rep]

x_train = train_df[sensor_cols].to_numpy()
y_train = train_df.label.to_numpy()

x_val = val_df[sensor_cols].to_numpy()
y_val = val_df.label.to_numpy()

x_val, x_test, y_val, y_test = train_test_split(
    x_val, y_val, test_size=0.5, stratify=y_val)

tuner = RandomSearch(
    Dense(),
    objective='val_accuracy',
    max_trials=10,
    directory='hp',
    seed=42,
    executions_per_trial=2,
    project_name=datetime.now().strftime("%Y%m%d-%H%M%S"))

""" tuner = Hyperband(
    Dense(),
    objective='val_accuracy',
    max_epochs=5,
    directory='hp',
    seed=42,
    executions_per_trial=2,
    project_name=datetime.now().strftime("%Y%m%d-%H%M%S")) """

""" tuner = BayesianOptimization(
    Dense(),
    objective='val_accuracy',
    max_trials=10,
    directory='hp',
    seed=42,
    executions_per_trial=2,
    project_name=datetime.now().strftime("%Y%m%d-%H%M%S")) """

tuner.search_space_summary()

tuner.search(x_train, y_train, batch_size=1024, epochs=10,
             validation_data=(x_val, y_val))

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.evaluate(x_test, y_test)
