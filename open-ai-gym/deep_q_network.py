import tensorflow as tf


class DeepQNetwork(tf.keras.Model):
    def __init__(self, n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3),
                                            strides=(2, 2),
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3),
                                            strides=(2, 2),
                                            activation='relu')
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x)
        return self.dense2(x)
