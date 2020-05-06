import tensorflow as tf
from kerastuner import HyperModel


class Model(HyperModel):
    """ When inheriting from base, method run_model must be
    overridden."""

    def run_model(self, batch_size, epochs):
        raise NotImplementedError("Method 'run_model' must be overridden!")

    def build(self, hp):
        raise NotImplementedError("Method 'build' must be overridden!")
