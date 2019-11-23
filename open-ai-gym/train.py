import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

import tensorflow as tf
import datetime as dt

train_writer = tf.summary.create_file_writer("./logs")

import gym
import numpy as np
from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork

env = gym.make('Breakout-v0')

n_actions = env.action_space.n
discount_factor = 0.99
update_interval = 25000
eps = 1
sample_size = 32

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
q = DeepQNetwork(n_actions)
q_hat = DeepQNetwork(n_actions)

# Initialize both networks with the same weights
q_hat.set_weights(q.get_weights())

memory = ReplayMemory(50_000)

optimizer = tf.keras.optimizers.RMSprop()


# epsilon - greedy algorithm
def select_a_with_epsilon_greedy(state, epsilon=eps):
    q_value = q(np.expand_dims(state / 255., axis=0)).numpy()
    action = np.argmax(q_value)
    if np.random.rand() < epsilon:
        action = np.random.randint(n_actions)
    return action


update_target_counter = 0


@tf.function
def train_decision_network():
    experience_samples = memory.sample_experiences(sample_size)
    previous_observations = np.array(
        [sample[0] for sample in experience_samples])
    updated_observations = np.array(
        [sample[3] for sample in experience_samples])
    rewards = np.array([sample[2] for sample in experience_samples])

    with tf.GradientTape() as tape:
        _q = tf.reduce_max(q_hat(updated_observations / 255.), axis=1)
        q_target = rewards + discount_factor * _q
        loss = tf.reduce_sum(tf.square(q_target - tf.reduce_max(q(previous_observations / 255.), axis=1)))
    grads = tape.gradient(loss, q.trainable_variables)
    optimizer.apply_gradients(zip(grads, q.trainable_variables))

    return loss

steps = 0

for i_episode in range(5000):

    print('Episode : {}'.format(i_episode + 1))

    if update_target_counter > 0 and update_target_counter == update_interval:
        q_hat.set_weights(q.get_weights())
        eps = eps - 0.01 if eps > 0.02 else 0.01
        update_target_counter = 0
        print('Target network updated, eps : {}'.format(eps))

    observation = env.reset()
    for t in range(100):
        experience = [None] * 4
        experience[0] = observation  # previous observation
        #env.render()
        action = select_a_with_epsilon_greedy(observation, eps)
        observation, reward, done, info = env.step(action)
        experience[1] = action
        experience[2] = reward
        experience[3] = observation  # next observation
        if reward > 0:
            memory.add_experience(experience)
        if memory.counter > 100:
            loss = train_decision_network()
            update_target_counter += 1
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=steps)
                tf.summary.scalar('reward', reward, step=steps)
                steps += 1

        if done:
            print('Episode end!')
            break

env.close()

q.save_weights('q_weights.h5')

for ind in range(len(memory.experiences)):
    obs, act, r, _obs = memory.experiences[ind]
    if r > 0:
        print(obs.shape, act, r, _obs.shape)
