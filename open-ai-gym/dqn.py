import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import timeit

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    pass

from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import gym
import numpy as np
import cv2
import collections
import logging

logging.basicConfig(format='%(message)s', 
                    filename='dqn.log', level=logging.DEBUG)

env = gym.make('Pong-v0')
test_env = gym.make('Pong-v0')

N_ACTIONS = env.action_space.n
GAMMA = 0.99
MAX_EPISODE_LENGTH = 300
N_EPOSIDES = 10_000
UPDATE_INTERVAL = 2_500
REPLAY_MEMORY_SIZE = 125_000
STACK_SIZE = 4
IMG_HEIGHT = 84
IMG_WIDTH = 64
EPS_MAX = 1.0
EPS_MIN = 0.1
ANNEALATION_STEPS = 1_500_000
MIN_EXPERIENCE_STEPS = 10_000
MINI_BATCH_SIZE = 128

OPTMIZER = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

def get_current_epsilon(n_th_step):
    if n_th_step > ANNEALATION_STEPS:
        return EPS_MIN
    else:
        return EPS_MAX - (EPS_MAX - EPS_MIN) * n_th_step / ANNEALATION_STEPS

q = DeepQNetwork(N_ACTIONS)
q.build((None, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))

q_target = DeepQNetwork(N_ACTIONS)
q_target.build((None, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))

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
    
def preprocess_input(frames):
    observed = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))
    for ind, obs in enumerate(frames):
        # to gray and to 0-1
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (IMG_WIDTH, IMG_HEIGHT))
        observed[..., ind] += obs / 255.
    return observed.astype(np.float32)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(MINI_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE), dtype=tf.float32),
    tf.TensorSpec(shape=(MINI_BATCH_SIZE,), dtype=tf.uint8),
    tf.TensorSpec(shape=(MINI_BATCH_SIZE,), dtype=tf.float32),
    tf.TensorSpec(shape=(MINI_BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, STACK_SIZE), dtype=tf.float32),
    tf.TensorSpec(shape=(MINI_BATCH_SIZE,), dtype=tf.bool),
])
def perform_gradient_step_on_q_net(STATES, ACTIONS, REWARDS, NEXT_STATES, IS_TERMINALS):
 
    loss_fn = tf.keras.losses.Huber()
    
    targets = REWARDS + tf.multiply(tf.cast(IS_TERMINALS, tf.float32),
                                    GAMMA * tf.reduce_max(q_target(NEXT_STATES), axis=1))
    actions = tf.one_hot(ACTIONS, depth=N_ACTIONS)
           
    with tf.GradientTape() as tape:
        selected_action_values = tf.multiply(q(STATES), actions)
        selected_action_values = tf.reduce_max(selected_action_values, axis=1)
        objective = loss_fn(targets, selected_action_values)
        
    grads = tape.gradient(objective, q.trainable_weights)
    
    OPTMIZER.apply_gradients(zip(grads, q.trainable_weights))

def test_agent():
    
    rewards = []
    
    for ep in range(10):
    
        initial_frame = test_env.reset() # initial observation
        frames = collections.deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            frames.append(initial_frame)

        state = preprocess_input(frames)

        total_reward = 0

        for timestep in range(MAX_EPISODE_LENGTH):
            action = select_action_e_greedy(state, 0.05)
            
            frame, reward, terminal, info = test_env.step(action)
            
            frames.append(frame)
            
            state = preprocess_input(frames)
            
            total_reward += reward
                
            if terminal:
                break 
            
        rewards.append(total_reward)
            
    logging.info('Average total reward of 10 episodes : %.2f' % np.mean(rewards))

n_th_iteration = 0

for ep in range(N_EPOSIDES):
    initial_frame = env.reset() # initial observation
    frames = collections.deque(maxlen=STACK_SIZE)
    for _ in range(STACK_SIZE):
        frames.append(initial_frame)

    state = preprocess_input(frames)
    
    total_reward = 0
    
    start = timeit.timeit()
    
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
            samples = memory.sample_experiences(MINI_BATCH_SIZE)
            STATES = tf.convert_to_tensor([sample[0] for sample in samples], dtype=tf.float32)
            ACTIONS = tf.convert_to_tensor([sample[1] for sample in samples], dtype=tf.uint8)
            REWARDS = tf.convert_to_tensor([sample[2] for sample in samples], dtype=tf.float32)
            NEXT_STATES = tf.convert_to_tensor([sample[3] for sample in samples], dtype=tf.float32)
            IS_TERMINALS = tf.convert_to_tensor([sample[4] for sample in samples], dtype=tf.bool)
            
            perform_gradient_step_on_q_net(STATES,
                                            ACTIONS,
                                            REWARDS,
                                            NEXT_STATES,
                                            IS_TERMINALS)
            
            if (n_th_iteration - MIN_EXPERIENCE_STEPS) % UPDATE_INTERVAL == 0:
                logging.info('Iteration : %d | Updating target weights...' % n_th_iteration)
                
                q.save_weights('chkpt/q.h5')
                q_target.set_weights(q.get_weights())
                
                test_agent()
            
        if terminal:
            total_reward -= 1
            break
        
    end = timeit.timeit()
    print(start - end)
       
    logging.info(f'Iteration : {n_th_iteration} | Episode : {ep + 1} | Total reward : {total_reward}, episode length : {timestep}, current eps : {current_eps}')
    
env.close()