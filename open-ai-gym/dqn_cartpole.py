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

logging.basicConfig(format='%(message)s', 
                    filename='dqn_cartpole_v2.log', level=logging.DEBUG)

env = gym.make('CartPole-v0')
test_env = gym.make('CartPole-v0')

N_ACTIONS = env.action_space.n
GAMMA = 0.99
MAX_EPISODE_LENGTH = 300
N_EPOSIDES = 20_000
UPDATE_INTERVAL = 2_000
REPLAY_MEMORY_SIZE = 50_000
EPS_MAX = 1.0
EPS_MIN = 0.01
ANNEALATION_STEPS = 200_000
MIN_EXPERIENCE_STEPS = 1_000
MINI_BATCH_SIZE = 128
OBSERVATION_SIZE = 4
NUMBER_OF_TEST_EPISODES = 25

OPTMIZER = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

def get_current_epsilon(n_th_step):
    if n_th_step > ANNEALATION_STEPS:
        return EPS_MIN
    else:
        return EPS_MAX - (EPS_MAX - EPS_MIN) * n_th_step / ANNEALATION_STEPS

q = tf.keras.models.Sequential(layers=[
    Dense(32, activation='tanh', input_shape=(OBSERVATION_SIZE,)),
    Dense(64, activation='tanh'),
    Dense(N_ACTIONS),
])

q_target = tf.keras.models.Sequential(layers=[
    Dense(32, activation='tanh', input_shape=(OBSERVATION_SIZE,)),
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
def perform_gradient_step_on_q_net(STATES, ACTIONS, REWARDS, NEXT_STATES, IS_TERMINALS):
    loss_fn = tf.keras.losses.Huber()
    
    targets = REWARDS + tf.multiply(tf.cast(IS_TERMINALS, tf.float32),
                                    GAMMA * tf.reduce_max(q_target(NEXT_STATES), axis=1))
    actions = tf.one_hot(ACTIONS, depth=N_ACTIONS)
           
    with tf.GradientTape() as tape:
        selected_states = tf.multiply(q(STATES), actions)
        selected_states = tf.reduce_max(selected_states, axis=1)
        objective = loss_fn(targets, selected_states)
        
    grads = tape.gradient(objective, q.trainable_weights)
    
    OPTMIZER.apply_gradients(zip(grads, q.trainable_weights))
    
    return targets, selected_states, actions    

def test_agent():
    test_rewards = []
    
    for ep in range(NUMBER_OF_TEST_EPISODES):            
        test_obs = test_env.reset()
        test_state = preprocess_input(test_obs)
        
        total_test_reward = 0

        for timestep in range(MAX_EPISODE_LENGTH):
            test_action = select_action_e_greedy(test_state, 0.05)
            test_obs, test_reward, test_terminal, _ = test_env.step(test_action)
            test_state = preprocess_input(test_obs)
            total_test_reward += test_reward
        
            if test_terminal:
                break

        test_rewards.append(total_test_reward)
    
    logging.info(f'Mean total test reward over {NUMBER_OF_TEST_EPISODES} episodes : {np.mean(test_rewards)}')

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
        
        if terminal:
            reward = -1
        
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
            
            if n_th_iteration % UPDATE_INTERVAL == 0:
                logging.info('Iteration : %d | Updating target weights...' % n_th_iteration)
                q.save_weights('chkpt_cartpole/q.h5')
                q_target.set_weights(q.get_weights())

                test_agent()
                
                logging.info("Loss : %.5f" % np.mean((targets - selected_states)**2))
                    
        if terminal:
            break
    
    """
    if n_th_iteration % 500 == 0: 
        logging.info(f'Iteration : {n_th_iteration} | Episode : {ep + 1} | Total reward : {total_reward}, episode length : {timestep}, current eps : {current_eps}')        
    """

env.close()