import tensorflow as tf
import numpy as np
class Adapter(tf.layers.Layer):

class PNN_Adapter(tf.layers.Layer):
    def __init__():
        super(PNN_Adapter, self).__init__()
        self.scalar = tf.Variable(np.random.randn())
        self.layer = tf.keras.layers.Dense(32, activation='relu')


    #TODO: In this case, the output is 
    def computeOutputShape(self, inputShape):
        return [32,]

    #call() is where we do the computation.
    def call(self, input, kwargs):
        x = tf.multiply(self.scalar, input)
        return self.layer(x)

    #Every layer needs a unique name.
    def getClassName(self):
            return 'PNN_Adapter'


class PNN_Model(tf.keras.Model):
    def __init__(self, layers, pretrained=None):
        super(PNN_Model, self).__init__()
        self.layers = [l for l in layers]
        self.F = pretrained

    def get_pretrained_results(self, pretrained, inputs, result=[]):
        res = pretrained(inputs)
        if pretrained.pretrained is not None:
            return rec(pretrained.pretrained, inputs, result.append(res))
        else:
            return result.append(res)

    def call(self, inputs):
        y = inputs
        pretrained_results = get_pretrained_results(self, self.pretrained, y)
        for i in range(len(layers)):
            adapter 
            y = self.layers[i](y)

        return y


    model.outputs
class Adapter(tf.keras)
        


if __name__ == "__main__":
    pass