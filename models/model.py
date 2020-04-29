import tensorflow as tf


class Model:
    """ When inheriting from base, method run_model must be
    overridden."""

    def run_model(self):
        raise NotImplementedError("Method must be overridden!")

    def _early_stop(self):
        print('\nCreating callbacks...')
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.0001,
            patience=3)
