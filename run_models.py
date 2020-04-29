
import os
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import random
import statistics
from datasets import normalized_human_gestures as human_gestures
from utils.data_utils import fraction_train_test_split, feature_train_test_split
from models_config.pnn_config import PNN
from models_config.dense_config import old_dense_model
from models.pnn import PNN_Column, PNN_Model
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


def run_models(selected_models=[], reps=5, batch_size=1024, epoch=5):
    args = (reps, batch_size, epoch)

    models_config = {
        'old_dense': lambda: old_dense_model(*args),
        'pnn': lambda: PNN(*args)
    }

    for current_model in selected_models:

        model = models_config[current_model]()

        results = model.run_model()

        print(results)


if __name__ == "__main__":
    run_models(['pnn', 'old_dense'])
