import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import gym
import numpy as np
import cv2
import collections
import tensorflow as tf
import logging

logging.basicConfig(format='%(message)s', 
                    filename='dqn.log', level=logging.DEBUG)

env = gym.make('Breakout-v0')

N_ACTIONS = env.action_space.n
GAMMA = 0.999
MAX_EPISODE_LENGTH = 300
N_EPOSIDES = 5_000
UPDATE_INTERVAL = 5_000
REPLAY_MEMORY_SIZE = 50_000
STACK_SIZE = 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
EPS_MAX = 1.0
EPS_MIN = 0.1
ANNEALATION_STEPS = 500_000
MIN_EXPERIENCE_STEPS = 20_000
MINI_BATCH_SIZE = 32

OPTMIZER = tf.keras.optimizers.RMSprop(lr=1e-3)

def get_current_epsilon(n_th_step):
    if n_th_step > ANNEALATION_STEPS:
        return 0.1
    else:
        return EPS_MAX - (EPS_MAX - EPS_MIN) * n_th_step / ANNEALATION_STEPS

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
q = DeepQNetwork(N_ACTIONS)
q_target = DeepQNetwork(N_ACTIONS)

# Initialize both networks with the same weights
q.save_weights('chkpt/q_weights')
q_target.load_weights('chkpt/q_weights')

memory = ReplayMemory(REPLAY_MEMORY_SIZE)

def select_action_e_greedy(state, current_eps):
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
        observed[..., ind] += obs
    return observed.astype(np.float32)

@tf.function(experimental_relax_shapes=True)
def perform_gradient_step_on_q_net():
    targets = []
    states = []
    actions = []
    
    samples = memory.sample_experiences(MINI_BATCH_SIZE)
    
    for ind, experience_sample in enumerate(samples):
        state, action, reward, next_state, is_terminal = experience_sample
        
        if is_terminal:
            targets.append(-1)
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
    
    initial_frame = env.reset() # initial observation
    frames = collections.deque(maxlen=STACK_SIZE)
    for _ in range(STACK_SIZE):
        frames.append(initial_frame)

    state = preprocess_input(frames)
    
    total_reward = 0
    
    for timestep in range(MAX_EPISODE_LENGTH):
        
        n_th_iteration += 1
        current_eps = get_current_epsilon(n_th_iteration)
        
        action = select_action_e_greedy(state, current_eps)
        
        frame, reward, terminal, info = env.step(action)
        
        old_state = state.copy()
        
        frames.append(frame)
        
        state = preprocess_input(frames)
        
        experience = [old_state, action, reward, state, terminal]
 
        memory.add_experience(experience)
        
        total_reward += reward
        
        if n_th_iteration > MIN_EXPERIENCE_STEPS:
            targets, selected_states, actions = perform_gradient_step_on_q_net()
            
            if n_th_iteration % UPDATE_INTERVAL == 0:
                logging.info('Iteration : %d | Updating target weights...' % n_th_iteration)
                q.save_weights('chkpt/q_weights')
                q_target.load_weights('chkpt/q_weights')
            
        if terminal:
            total_reward -= 1
            
            logging.info(f'Iteration : {n_th_iteration} | Episode : {ep + 1} | Total reward : {total_reward}, episode length : {timestep}, current eps : {current_eps}')
            
            break        

env.close()