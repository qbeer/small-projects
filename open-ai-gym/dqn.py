import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf

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
MAX_EPISODE_LENGTH = 500
N_EPOSIDES = 10_000
UPDATE_INTERVAL = 1_000
REPLAY_MEMORY_SIZE = 200_000
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

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
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

@tf.function(experimental_relax_shapes=True)
def perform_gradient_step_on_q_net(STATES, ACTIONS, REWARDS, NEXT_STATES, IS_TERMINALS):
    targets = []
    states = []
    actions = []
    
    loss_fn = tf.keras.losses.Huber()
    
    for ind in range(MINI_BATCH_SIZE):
        state, action, reward, next_state, is_terminal = STATES[ind], ACTIONS[ind],\
            REWARDS[ind], NEXT_STATES[ind], IS_TERMINALS[ind]
        
        if not is_terminal:
            bootstrap = GAMMA * tf.reduce_max(q_target(tf.expand_dims(next_state, axis=0)))
            reward += bootstrap
            
        targets.append(reward)
        states.append(state)
        actions.append(tf.one_hot(action, depth=N_ACTIONS))
    
    states = tf.stack(states)
    
    actions = tf.stack(actions)
    
    targets = tf.stack(targets)
           
    with tf.GradientTape() as tape:
        selected_states = tf.multiply(q(states), actions)
        selected_states = tf.reduce_max(selected_states, axis=1)
        objective = loss_fn(targets, selected_states)
        
    grads = tape.gradient(objective, q.trainable_weights)
    
    OPTMIZER.apply_gradients(zip(grads, q.trainable_weights))
    
    return targets, selected_states, actions

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
            STATES = tf.convert_to_tensor([sample[0] for sample in samples])
            ACTIONS = tf.convert_to_tensor([sample[1] for sample in samples])
            REWARDS = tf.convert_to_tensor([sample[2] for sample in samples])
            NEXT_STATES = tf.convert_to_tensor([sample[3] for sample in samples])
            IS_TERMINALS = tf.convert_to_tensor([sample[4] for sample in samples])
            
            targets, selected_states, actions = perform_gradient_step_on_q_net(STATES,
                                                                               ACTIONS,
                                                                               REWARDS,
                                                                               NEXT_STATES,
                                                                               IS_TERMINALS)
            
            if (n_th_iteration - MIN_EXPERIENCE_STEPS) % UPDATE_INTERVAL == 0:
                logging.info('Iteration : %d | Updating target weights...' % n_th_iteration)
                q.save_weights('chkpt/q.h5')
                
                logging.info('Loss : %.5f' % np.mean((targets - selected_states)**2))
                
                for t, s_s in zip(targets, selected_states):
                    logging.info(f'target : {t}, predicted_target : {s_s}')
                
                q_target.set_weights(q.get_weights())
                
                test_agent()
                
                samples = memory.sample_experiences(16)
                
                for samp in samples:
                    logging.info(f'action : {samp[1]}, reward : {samp[2]}, terminal : {samp[-1]}')
            
        if terminal:
            total_reward -= 1
            break
    
    if n_th_iteration % 3000 == 0:        
        logging.info(f'Iteration : {n_th_iteration} | Episode : {ep + 1} | Total reward : {total_reward}, episode length : {timestep}, current eps : {current_eps}')      

env.close()