import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import gym
import numpy as np
import cv2
import collections
import tensorflow as tf
import logging

env = gym.make('Pong-v0')

N_ACTIONS = env.action_space.n
MAX_EPISODE_LENGTH = 500
STACK_SIZE = 4
IMG_HEIGHT = 84
IMG_WIDTH = 64

q = DeepQNetwork(N_ACTIONS)
q.build((None, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))
q.load_weights('chkpt/q.h5')

def select_action_e_greedy(state, current_eps=0.05):
    eps = np.random.uniform()
    if eps < current_eps:
        return np.random.randint(0, N_ACTIONS)
    else:
        state = np.expand_dims(state, axis=0) # batch size of 1
        action = np.argmax(q(state))
        return np.squeeze(action, axis=0) # reduce back to a scalar
    
def preprocess_input(frames):
    observed = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))
    for ind, obs in enumerate(frames):
        # to gray and to 0-1
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (IMG_WIDTH, IMG_HEIGHT))
        observed[..., ind] += obs / 255.
    return observed.astype(np.float32)

for ep in range(10):
    
    initial_frame = env.reset() # initial observation
    frames = collections.deque(maxlen=STACK_SIZE)
    for _ in range(STACK_SIZE):
        frames.append(initial_frame)

    state = preprocess_input(frames)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):

        env.render()
        
        action = select_action_e_greedy(state)
        print(action, env.unwrapped.get_action_meanings()[action])
        
        frame, reward, terminal, info = env.step(action)
        
        old_state = state.copy()
        
        frames.append(frame)
        
        state = preprocess_input(frames)
        
        total_reward += reward
            
        if terminal:
            break
    
    print(f'Total reward : {total_reward}')
    
print('\nCompletely random agent : ')    

for ep in range(10):
    
    initial_frame = env.reset() # initial observation
    frames = collections.deque(maxlen=STACK_SIZE)
    for _ in range(STACK_SIZE):
        frames.append(initial_frame)

    state = preprocess_input(frames)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):

        env.render()
        
        action = env.action_space.sample()
        
        frame, reward, terminal, info = env.step(action)
        
        old_state = state.copy()
        
        frames.append(frame)
        
        state = preprocess_input(frames)
        
        total_reward += reward
            
        if terminal:
            break
    
    print(f'Total reward : {total_reward}')

env.close()