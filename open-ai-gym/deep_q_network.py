import tensorflow as tf


def DeepQNetwork(n_actions):
    conv1 = tf.keras.layers.Conv2D(32, (3, 3),
                                        strides=(2, 2),
                                        activation='relu')
    conv2 = tf.keras.layers.Conv2D(64, (3, 3),
                                        activation='relu')
    conv3 = tf.keras.layers.Conv2D(128, (3, 3),
                                        strides=(2, 2),
                                        activation='relu')
    conv4 = tf.keras.layers.Conv2D(256, (3, 3),
                                        activation='relu')
    pool = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(128, activation='relu') 
    dense2 = tf.keras.layers.Dense(n_actions)
    
    model = tf.keras.models.Sequential(layers=[
        conv1, conv2, conv3, conv4, pool, dense1, dense2
    ])
    
    model.build((None, 84, 84, 4))
    
    return model
