import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Harmonics(Layer):
    def __init__(self, units=32):
        """tf layer that that takes one input and creates a layer that
        contains sin and cos activations with the harmonics of omega
        where omega is the only trainable variable

        Args:
          units: number of neurons
        """
        super(Harmonics, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        omega_init = tf.constant(
            np.pi, dtype=tf.float32
        )  # initialise omega as pi (from the physics)
        self.omega = tf.Variable(initial_value=omega_init, trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        self.w = self.omega * tf.range(1, self.units + 1, 1, dtype=tf.float32)
        return tf.concat([tf.sin(self.w * inputs), tf.cos(self.w * inputs)], axis=1)
