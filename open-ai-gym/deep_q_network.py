import tensorflow as tf


def DeepQNetwork(n_actions):
    conv1 = tf.keras.layers.Conv2D(64, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    conv2 = tf.keras.layers.Conv2D(256, (3, 3),
                                        activation='relu',
                                        kernel_initializer='RandomNormal')
    flat = tf.keras.layers.Flatten()
    dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='RandomNormal') 
    dense2 = tf.keras.layers.Dense(n_actions)
    
    model = tf.keras.models.Sequential(layers=[
        conv1, conv2, flat, dense1, dense2
    ])
    
    model.build((None, 64, 48, 4))
    
    return model
