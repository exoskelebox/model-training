import tensorflow as tf


class Model:
    """ When inheriting from base, method run_model must be
    overridden."""

    def run_model(self):
        raise NotImplementedError("Method must be overridden!")

    def _compile_model(self, model):
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print('Model compiled.')

    def _early_stop(self):
        print('Creating callbacks...')
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', min_delta=0.0001,
            patience=3)
