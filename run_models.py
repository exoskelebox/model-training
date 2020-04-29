import tensorflow as tf
import os
import logging
from models import Dense

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


def run_models(selected_models=[], reps=5, batch_size=1024, epoch=5):

    args = (reps, batch_size, epoch)

    models_config = {
        'dense': lambda: Dense(*args)
    }

    for current_model in selected_models:

        model = models_config[current_model]()

        results = model.run_model()

        print(results)


if __name__ == "__main__":
    run_models(['dense'])
