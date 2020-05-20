import os
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # nopep8
os.environ["KMP_AFFINITY"] = "noverbose"  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import tensorflow as tf
from models import Dense, run_pnn, combined_pnn, fine_tuning, fn_pnn

tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(3)

# create results logger
logger = logging.getLogger('results')
logger.setLevel(logging.INFO)

# create file handler
fh = logging.FileHandler('results.log')
fh.setLevel(logging.INFO)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def run_models(selected_models=[], batch_size=1024, epochs=5):

    models_config = {
        'dense': Dense,
        'pnn': run_pnn.PNN,
        'cpnn': combined_pnn.Combined_PNN,
        'finetuning': fine_tuning.FineTuned,
        'fn_pnn': fn_pnn.PNN
    }

    for current_model in selected_models:

        model = models_config[current_model]()

        results = model.run_model(batch_size, epochs)

        logger.info((current_model, *results))


if __name__ == "__main__":
    run_models(['cpnn'], 1024, 100)
