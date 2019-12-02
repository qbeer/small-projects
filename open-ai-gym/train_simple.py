import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import gym
import numpy as np
import cv2
from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
import collections

train_writer = tf.summary.create_file_writer("./logs")

env = gym.make('Breakout-v0')

n_actions = env.action_space.n
discount_factor = 0.99
update_interval = 5000
EPS_MAX = 1
EPS_MIN = 0.1

eps = EPS_MAX

EPS_ANNEALING_INTERVAL = 100000

sample_size = 64

HEIGHT = 60
WIDTH = 48
N_STACK = 4

MIN_OBSERVATIONS = 2500

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
q = DeepQNetwork(n_actions)
q_hat = DeepQNetwork(n_actions)

# Initialize both networks with the same weights
q_hat.set_weights(q.get_weights())

memory = ReplayMemory(5000)

optimizer = tf.keras.optimizers.RMSprop()


def preprocess_input(observations):
    observed = []
    for obs in observations:
        # to gray and to 0-1
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (WIDTH, HEIGHT))
        observed.append(obs / 255.)
    observed = np.array(observed)
    observed = np.transpose(observed, (1, 2, 0))
    return observed


# epsilon - greedy algorithm
def select_a_with_epsilon_greedy(state, epsilon=eps):
    state = preprocess_input([state] * N_STACK)
    q_value = q(np.expand_dims(state, axis=0)).numpy()
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

    done = np.array([sample[4] for sample in experience_samples])

    with tf.GradientTape() as tape:
        _q = tf.reduce_max(q_hat(updated_observations), axis=1)
        q_target = rewards + discount_factor * _q
        loss = tf.reduce_sum(
            tf.square(q_target -
                      tf.reduce_max(q(previous_observations), axis=1)))
    grads = tape.gradient(loss, q.trainable_variables)
    optimizer.apply_gradients(zip(grads, q.trainable_variables))

    return loss


steps = 0

for i_episode in range(5000):

    print('Episode : {}, memory counter : {}'.format(i_episode + 1,
                                                     memory.counter))

    if update_target_counter > 0 and update_target_counter == update_interval:
        q_hat.set_weights(q.get_weights())
        update_target_counter = 0

    observation = env.reset()

    observation1 = collections.deque(maxlen=N_STACK)
    observation2 = collections.deque(maxlen=N_STACK)

    observation1.append(observation)

    for t in range(500):
        action = select_a_with_epsilon_greedy(observation, eps)

        env.render()

        observation, reward, done, info = env.step(action)
        observation1.append(observation)
        observation2.append(observation)

        if len(observation1) == 4 and len(observation2) == 4:
            experience = [None] * 5
            experience[0] = preprocess_input(
                observation1)  # previous observation
            experience[1] = action
            experience[2] = reward
            experience[3] = preprocess_input(observation2)  # next observation
            experience[4] = done  # store done to add negative reward on death
            memory.add_experience(experience)
        if memory.counter > MIN_OBSERVATIONS:
            loss = train_decision_network()
            update_target_counter += 1
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=steps)
                tf.summary.scalar('reward', reward, step=steps)
                tf.summary.scalar('eps', eps, step=steps)
                steps += 1

        if done:
            print('Episode end!')
            break

        eps -= (EPS_MAX - EPS_MIN) / EPS_ANNEALING_INTERVAL

env.close()

q.save_weights('q_weights.h5')

for ind in range(len(memory.experiences)):
    obs, act, r, _obs = memory.experiences[ind]
    if r > 0:
        print(obs.shape, act, r, _obs.shape)
