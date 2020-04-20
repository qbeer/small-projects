import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import gym
import numpy as np
import cv2
import collections
import tensorflow as tf
from tensorflow.keras.layers import Dense
import logging

logging.basicConfig(format='%(message)s', 
                    filename='dqn_cartpole.log', level=logging.DEBUG)

env = gym.make('CartPole-v0')

N_ACTIONS = env.action_space.n
GAMMA = 0.99
MAX_EPISODE_LENGTH = 300
N_EPOSIDES = 50_000
UPDATE_INTERVAL = 2000
REPLAY_MEMORY_SIZE = 500_000
EPS_MAX = 1.0
EPS_MIN = 0.05
ANNEALATION_STEPS = 1_000_000
MIN_EXPERIENCE_STEPS = 75_000
MINI_BATCH_SIZE = 128
OBSERVATION_SIZE = 4

OPTMIZER = tf.keras.optimizers.RMSprop(lr=1e-3)

def get_current_epsilon(n_th_step):
    if n_th_step > ANNEALATION_STEPS:
        return 0.1
    else:
        return EPS_MAX - (EPS_MAX - EPS_MIN) * n_th_step / ANNEALATION_STEPS

q = tf.keras.models.Sequential(layers=[
    Dense(32, activation='tanh', input_shape=(None, OBSERVATION_SIZE)),
    Dense(64, activation='tanh'),
    Dense(N_ACTIONS),
])
q_target = tf.keras.models.Sequential(layers=[
    Dense(32, activation='tanh', input_shape=(None, OBSERVATION_SIZE)),
    Dense(64, activation='tanh'),
    Dense(N_ACTIONS),
])

# Initialize both networks with the same weights
q_target.set_weights(q.get_weights())

memory = ReplayMemory(REPLAY_MEMORY_SIZE)

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

@tf.function(experimental_relax_shapes=True)
def perform_gradient_step_on_q_net():
    targets = []
    states = []
    actions = []
    
    samples = memory.sample_experiences(MINI_BATCH_SIZE)
    
    for ind, experience_sample in enumerate(samples):
        state, action, reward, next_state, is_terminal = experience_sample
        
        if is_terminal:
            targets.append(-1.)
        else:
            bootstrapped_reward = reward
            bootstrap = GAMMA * tf.reduce_max(q_target(tf.expand_dims(next_state, axis=0)))
            bootstrapped_reward += bootstrap
            targets.append(bootstrapped_reward)
           
        states.append(state)
        actions.append(tf.one_hot(action, depth=N_ACTIONS))
    
    states = tf.stack(states)
    
    actions = tf.stack(actions)
    
    targets = tf.stack(targets)
           
    with tf.GradientTape() as tape:
        selected_states = tf.multiply(q(states), actions)
        selected_states = tf.reduce_max(selected_states, axis=1)
        objective = tf.reduce_mean((targets -  selected_states)**2)
        
    grads = tape.gradient(objective, q.trainable_weights)
    
    OPTMIZER.apply_gradients(zip(grads, q.trainable_weights))
    
    return targets, selected_states, actions    

n_th_iteration = 0

for ep in range(N_EPOSIDES):
    
    obs = env.reset() # initial observation

    state = preprocess_input(obs)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):
        
        n_th_iteration += 1
        current_eps = get_current_epsilon(n_th_iteration)
        
        action = select_action_e_greedy(state, current_eps)
        
        obs, reward, terminal, info = env.step(action)
        
        old_state = state.copy()
        
        state = preprocess_input(obs)
        
        experience = [old_state, action, reward, state, terminal]
 
        memory.add_experience(experience)
        
        total_reward += reward
        
        if n_th_iteration > MIN_EXPERIENCE_STEPS:
            targets, selected_states, actions = perform_gradient_step_on_q_net()
            
            if n_th_iteration % UPDATE_INTERVAL == 0:
                logging.info('Iteration : %d | Updating target weights...' % n_th_iteration)
                q.save_weights('chkpt_cartpole/q.h5')
                q_target.set_weights(q.get_weights())
                
                logging.info(np.mean((targets - selected_states)**2))
            
        if terminal:
            total_reward -= 1
            break
            
    logging.info(f'Iteration : {n_th_iteration} | Episode : {ep + 1} | Total reward : {total_reward}, episode length : {timestep}, current eps : {current_eps}')        

env.close()