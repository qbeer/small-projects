import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import gym
import numpy as np
from deep_q_network import DeepQNetwork

env = gym.make('Breakout-v0')

n_actions = env.action_space.n
discount_factor = 0.95
update_interval = 100
eps = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2),
                                 activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2),
                                 activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(n_actions))

model.build(input_shape=(1, 210, 160, 3))

model.load_weights('q_weights.h5')

q = model


# epsilon - greedy algorithm
def select_a_with_epsilon_greedy(state, epsilon=eps):
    q_value = q(np.expand_dims(state / 255., axis=0)).numpy()
    action = np.argmax(q_value)
    if np.random.rand() < epsilon:
        action = np.random.randint(4)
    return action


for i_episode in range(12):

    print('Episode : {}'.format(i_episode + 1))

    observation = env.reset()
    for t in range(100):
        experience = [None] * 4
        experience[0] = observation  # previous observation
        env.render()
        action = select_a_with_epsilon_greedy(observation, epsilon=0.01)
        observation, reward, done, info = env.step(action)
        experience[1] = action
        experience[2] = reward
        experience[3] = observation  # next observation

        print(reward)

        if done:
            break

env.close()