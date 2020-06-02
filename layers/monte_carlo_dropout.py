import tensorflow as tf

class MCDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)