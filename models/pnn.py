from __future__ import absolute_import
import tensorflow as tf
import numpy as np


class PNN_Adapter(tf.keras.layers.Layer):
    def __init__(self, units=2, activation='relu'):
        super(PNN_Adapter, self).__init__()
        self.units=units
        self.activation=activation
    
    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,), 
                                initializer='random_normal',
                                trainable=True)
        self.dense = tf.keras.layers.Dense(self.units, activation=self.activation)

    #call() is where we do the computation.
    def call(self, inputs):
        x = tf.multiply(inputs, self.a)
        x = self.dense(x)
        return x
        
class PNN_Model(tf.keras.Model):
    def __init__(self, layer_info, previous=None):
        super(PNN_Model, self).__init__()
        self._layerinfo=layer_info
        self.layer_outputs=[None for _ in layer_info]
        self.previous=previous

    def build(self, input_shape):
        self.core_layers = [layer_info['type'](units=layer_info['units'], activation=layer_info['activation']) for layer_info in self._layerinfo]
        if self.previous:
            all_adapter_layers = []
            for pretrained in self.previous:
                pretrained.trainable = False
                model_adapter_layers = []
                for layer in self._layerinfo[:-1]: # last layer doesnt get used for new model
                    adapter = PNN_Adapter()
                    adapter.build(layer['units'])
                    model_adapter_layers.append(adapter)
                if len(model_adapter_layers) >= len(self.core_layers):
                    print(f"ERROR: too many adapter layers. Needs {len(self.core_layers)-1}, has {len(model_adapter_layers)}")
                all_adapter_layers.append(model_adapter_layers)
            self.adapter_layers = all_adapter_layers
        #self.outputs = [l.output for l in self.core_layers]

    def call(self, inputs):
        y = inputs
        pretrained_outputs = []
        if self.previous:
            for pretrained in self.previous:
                pretrained(y)
                pretrained_outputs.append(pretrained.layer_outputs)

        for i in range(len(self.core_layers)):
            if self.previous and i >= 1: # First layer doesnt get input from pretrained models
                adapted_pretrained_outputs = [self.adapter_layers[j][i-1](pretrained_outputs[j][i-1]) for j in range(len(pretrained_outputs))]
                y = tf.keras.layers.concatenate([y, *adapted_pretrained_outputs])

            y = self.core_layers[i](y)
            self.layer_outputs[i] = y
        return y

def test_adapter():
    x = tf.ones((5, 5))
    adapter = PNN_Adapter(units=5)
    print(x)
    y = adapter(x)
    print(y.numpy())
    print(adapter.weights)

def test_layered():
    x = tf.ones((2, 2))
    layers = [{'type':tf.keras.layers.Dense, 'units':2, 'activation':'relu'} for i in range(3)]
    old_model = PNN_Model(layers)
    old_model.build(x.shape)
    model = PNN_Model(layers, [old_model])

    #model.build(x.shape)
    #model.outputs=[l.output for l in model.core_layers]
    print(f"Inputs:\n{x}")
    y = model(x)
    print(f"Output:\n {y.numpy()}")
    print(f"Outputs:\n {model.layer_outputs}")
    #print(f"Weigths:\n {model.weights}")
    print(f"Summary:")
    model.summary()    
    print(f"Summary: Old_Model")
    old_model.summary()

if __name__ == "__main__":
    test_layered()