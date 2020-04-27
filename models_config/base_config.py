import tensorflow as tf

class base:
    """ When inheriting from base, method run_model must be
    overridden."""
    
    # def __init__(self):
    #     pass
    
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


if __name__ == "__main__":
    base_01 = base()
    base_01.run_model()