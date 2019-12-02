import tensorflow as tf


class DeepQNetwork(tf.keras.Model):
    def __init__(self, n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(4, (3, 3),
                                            strides=(2, 2),
                                            activation='relu',
                                            input_shape=(32, 84, 60, 4))
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3),
                                            strides=(2, 2),
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
