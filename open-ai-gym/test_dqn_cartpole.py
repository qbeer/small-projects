import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import gym
import numpy as np
import cv2
import collections
import tensorflow as tf
from tensorflow.keras.layers import Dense
import logging

env = gym.make('CartPole-v0')

N_ACTIONS = env.action_space.n
MAX_EPISODE_LENGTH = 2500
OBSERVATION_SIZE = 4

q = tf.keras.models.Sequential(layers=[
    Dense(32, activation='tanh', input_shape=(OBSERVATION_SIZE,)),
    Dense(64, activation='tanh'),
    Dense(N_ACTIONS),
])

q.load_weights('chkpt_cartpole/q.h5')

def select_action_e_greedy(state, current_eps):
    eps = np.random.uniform()
    if eps < current_eps:
        return np.random.randint(0, N_ACTIONS)
    else:
        state = np.expand_dims(state, axis=0) # batch size of 1
        action = np.argmax(q(state))
        return np.squeeze(action, axis=0) # reduce back to a scalar
    
def preprocess_input(observation):
    return observation.reshape(OBSERVATION_SIZE).astype(np.float32)

for ep in range(10):
    obs = env.reset() # initial observation
    state = preprocess_input(obs)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):
        #env.render()   
        action = select_action_e_greedy(state, 0.05)
        #action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
        state = preprocess_input(obs)
        total_reward += reward
        if terminal:
            break
    
    print(f'Total reward : {total_reward}') 
   
print('Completely random agent:')
    
for ep in range(10):
    obs = env.reset() # initial observation
    state = preprocess_input(obs)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):
        #env.render()   
        #action = select_action_e_greedy(state, 0.05)
        action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
        state = preprocess_input(obs)
        total_reward += reward
        if terminal:
            break
    
    print(f'Total reward : {total_reward}')        

env.close()