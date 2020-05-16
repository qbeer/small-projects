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
        conv1, conv1_2, conv2, conv2_2, pool, dense1, dense2, out
    ])
    
    model.build((None, 84, 64, 4))
    
    return model
