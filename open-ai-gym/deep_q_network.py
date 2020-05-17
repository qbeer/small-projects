import tensorflow as tf


def DeepQNetwork(n_actions):
    conv1 = tf.keras.layers.Conv2D(32, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    conv1_2 = tf.keras.layers.Conv2D(64, (3, 3),
                                        strides=2,
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    conv2 = tf.keras.layers.Conv2D(64, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    conv2_2 = tf.keras.layers.Conv2D(128, (3, 3),
                                        strides=2,
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    pool = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='RandomNormal') 
    dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='RandomNormal')
    out = tf.keras.layers.Dense(n_actions, kernel_initializer='RandomNormal')
    
    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(32, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
        tf.keras.layers.Conv2D(64, (3, 3),
                                            strides=2,
                                            activation='relu',
                                            kernel_initializer='RandomNormal')
        tf.keras.layers.Conv2D(64, (3, 3),
                                            activation='relu',
                                            kernel_initializer='RandomNormal')
        tf.keras.layers.Conv2D(128, (3, 3),
                                            strides=2,
                                            activation='relu',
                                            kernel_initializer='RandomNormal')
        tf.keras.layers.Flatten()
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='RandomNormal') 
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='RandomNormal')
        tf.keras.layers.Dense(n_actions, kernel_initializer='RandomNormal')
    ])
    
    return model
