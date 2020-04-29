from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from typing import List


class PNN_Adapter(tf.keras.layers.Layer):
    def __init__(self, units=2, activation='relu'):
        super(PNN_Adapter, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,),
                                 initializer='random_normal',
                                 trainable=True)
        self.dense = tf.keras.layers.Dense(
            self.units, activation=self.activation)

    def call(self, inputs):
        x = tf.multiply(inputs, self.a)
        x = self.dense(x)
        return x


class PNN_Column(tf.keras.Model):
    def __init__(self, adapter_layers: List[tf.keras.layers.Layer], core_layers: List[tf.keras.layers.Layer], generation: int = 0):
        super(PNN_Column, self).__init__()
        self.core_layers = core_layers
        self.adapter_layers = adapter_layers
        self.generation = generation
        self.layer_outputs = []

    def call(self, inputs):
        core_layers = enumerate(self.core_layers)
        i, core_layer = next(core_layers)
        x = core_layer(inputs[i])
        y = [x]
        for i, core_layer in core_layers:
            if self.generation >= 1:
                k = [inputs[j+1][i-1] for j in range(self.generation)]
                for adapter_layer in iter(self.adapter_layers):
                    k = adapter_layer(k)
                x = tf.keras.layers.concatenate(
                    [x, *k])
                x = core_layer(x)
            else:
                x = core_layer(x)
            y.append(x)

        return y


class PNN_Model(tf.keras.Model):
    def __init__(self, feature_layer=lambda x: x, columns=[]):
        super(PNN_Model, self).__init__()
        self.columns = columns
        self.feature_layer = feature_layer
        for column in self.columns[:-1]:
            # Freeze all but the last column
            column.trainable = False

    def call(self, inputs):
        y = [self.feature_layer(inputs)]
        for column in self.columns:
            y.append(column(y))
        # return the output of the last layer of the last column
        return y[-1][-1]


def test_adapter():
    x = tf.ones((5, 5))
    adapter = PNN_Adapter(units=2)
    print(x)
    y = adapter(x)
    print(y.numpy())
    print(adapter.weights)


# def test_layered():
#     x = tf.ones((2, 2))
#     adapter = [tf.keras.layers.Dense(16, 'relu')]
#     core = [tf.keras.layers.Dense(
#         64, 'relu'), tf.keras.layers.Dense(64, 'relu'), tf.keras.layers.Dense(1, 'softmax')]

#     column_0 = PNN_Column(adapter, core)
#     model_0 = PNN_Model(columns=[column_0])
#     model_0.build(x.shape)
#     model_0.summary()

#     column_1 = PNN_Column(adapter, core, generation=1)
#     model_1 = PNN_Model(columns=[column_0, column_1])
#     model_1.build(x.shape)
#     model_1.summary()
    """


    # model.build(x.shape)
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
    column_1.summary() """


if __name__ == "__main__":
    # print("HELLO=!")
    test_adapter()
    # test_layered()
