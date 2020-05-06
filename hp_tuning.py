import os
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # nopep8
os.environ["KMP_AFFINITY"] = "noverbose"  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import tensorflow as tf
from datasets import normalized_human_gestures as human_gestures
from kerastuner.oracles import RandomSearch
from models import Dense
from tuners import KFoldTuner


tuner = KFoldTuner(
    oracle=RandomSearch(
        objective='val_accuracy',
        max_trials=5
    ),
    hypermodel=Dense(),
    directory='hp',
    project_name='dense')

tuner.search_space_summary()

tuner.search(epochs=5, batch_size=1024)

tuner.results_summary()
