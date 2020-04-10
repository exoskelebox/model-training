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


class PNN_Column(tf.keras.Model):
    def __init__(self, layer_info:dict, generation:int=0):
        super(PNN_Column, self).__init__()
        self._layerinfo=layer_info
        self.layer_outputs=[None for _ in layer_info['core']]
        self.generation = generation

    def build(self, input_shape):
        core_info =  self._layerinfo['core']
        self.core_layers = [layer_info['type'](
            units=layer_info['units'], 
            activation=layer_info['activation']) 
            for layer_info in core_info]

        adapter_info = self._layerinfo['adapters']
        self.adapter_layers = [[None for j in range(len(self.core_layers)-1)] 
        # First layer doesnt get input from pretrained models
            for i in range(self.generation)]
        
        for i in range(self.generation):
            for j in range(len(self.core_layers)-1):
                # First layer doesnt get input from pretrained models
                adapter = adapter_info['type'](
                    units=adapter_info['units'], 
                    activation=adapter_info['activation'])
                adapter.build(core_info[j]['units'])
                self.adapter_layers[i][j] = adapter

    def call(self, inputs):
        y = inputs[0]
        pretrained_outputs = inputs[1:]
        for i in range(len(self.core_layers)):
            if self.generation >= 1 and i >= 1: 
                # First layer doesnt get input from pretrained models
                adapted_pretrained_outputs = [self.adapter_layers[j][i-1](
                    pretrained_outputs[j][i-1]) for j in range(self.generation)]
                y = tf.keras.layers.concatenate([y, *adapted_pretrained_outputs])
            y = self.core_layers[i](y)
            self.layer_outputs[i] = y
        return self.layer_outputs


class PNN_Model(tf.keras.Model):
    def __init__(self, feature_layer=lambda x: x, columns=[]):
        super(PNN_Model, self).__init__()
        self.columns=columns
        self.feature_layer = feature_layer
        self.column_outputs = None
        for column in self.columns[:-1]:
            # Freeze all but the last column
            column.trainable = False

    def call(self, inputs):
        y = [self.feature_layer(inputs)]
        for column in self.columns:
            y.append(column(y))
        self.column_outputs = y
        return y[-1][-1] # return the output of the last layer of the last column
        

def test_adapter():
    x = tf.ones((5, 5))
    adapter = PNN_Adapter(units=5)
    print(x)
    y = adapter(x)
    print(y.numpy())
    print(adapter.weights)

def test_layered():
    x = tf.ones((2, 2))
    adapters = {
        'type':tf.keras.layers.Dense, 
        'units':3, 
        'activation':'relu'} 
    core = [{
        'type':tf.keras.layers.Dense, 
        'units':5, 
        'activation':'relu'} 
        for i in range(3)]
    layer_info = {'core':core, 'adapters':adapters}

    column_0 = PNN_Column(layer_info)
    model_0 = PNN_Model(columns=[column_0])
    model_0.build(x.shape)


    column_1 = PNN_Column(layer_info, generation=1)
    model_1 = PNN_Model(columns=[column_0, column_1])
    model_1.build(x.shape)

    #model.build(x.shape)
    #model.outputs=[l.output for l in model.core_layers]
    print(f"Inputs:\n{x}")
    y = model_1(x)
    print(f"Output:\n {y.numpy()}")
    print(f"Outputs:\n {model_1.column_outputs}")
    #print(f"Weigths:\n {model.weights}")
    print(f"Summary: model_0")
    model_0.summary()
    print(f"Summary: model_1")
    model_1.summary()    
    print(f"Summary: column_0")
    column_0.summary()    
    print(f"Summary: column_1")
    column_1.summary()    

if __name__ == "__main__":
    test_layered()