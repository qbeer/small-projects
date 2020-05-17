import tensorflow as tf


def DeepQNetwork(n_actions):
    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(32, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal'),
        tf.keras.layers.Conv2D(64, (3, 3),
                                            strides=2,
                                            activation='relu',
                                            kernel_initializer='RandomNormal'),
        tf.keras.layers.Conv2D(64, (3, 3),
                                            activation='relu',
                                            kernel_initializer='RandomNormal'),
        tf.keras.layers.Conv2D(128, (3, 3),
                                            strides=2,
                                            activation='relu',
                                            kernel_initializer='RandomNormal'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='RandomNormal') ,
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='RandomNormal'),
        tf.keras.layers.Dense(n_actions, kernel_initializer='RandomNormal')
    ])
    
    return model
