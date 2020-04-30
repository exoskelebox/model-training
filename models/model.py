import tensorflow as tf


class Model:
    """ When inheriting from base, method run_model must be
    overridden."""

    def run_model(self):
        raise NotImplementedError("Method must be overridden!")
