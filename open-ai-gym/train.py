import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import datetime as dt

train_writer = tf.summary.create_file_writer("./logs")

import gym
import numpy as np
import cv2
from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork

env = gym.make('Breakout-v0')

n_actions = env.action_space.n
discount_factor = 0.9
update_interval = 25000
EPS_MAX = 1
EPS_MIN = 0.1

eps = EPS_MAX

EPS_ANNEALING_INTERVAL = 1000000

sample_size = 32

HEIGHT = 84
WIDTH = 60

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
q = DeepQNetwork(n_actions)
q_hat = DeepQNetwork(n_actions)

# Initialize both networks with the same weights
q_hat.set_weights(q.get_weights())

memory = ReplayMemory(50000)

optimizer = tf.keras.optimizers.RMSprop()


def preprocess_input(observations):
    observed = []
    for obs in observations:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        observed.append(cv2.resize(obs, (HEIGHT, WIDTH)))
    observed = np.array(observed)
    return observed.reshape((HEIGHT, WIDTH, 4))


# epsilon - greedy algorithm
def select_a_with_epsilon_greedy(state, epsilon=eps):
    state = preprocess_input(state)
    state = [state] * sample_size
    state = np.array(state).reshape(sample_size, HEIGHT, WIDTH, 4) / 255.
    q_value = q(state).numpy()
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

    print(updated_observations.shape)
    print(previous_observations.shape)

    with tf.GradientTape() as tape:
        _q = tf.reduce_max(q_hat(updated_observations / 255.), axis=1)
        print(_q)
        q_target = rewards + discount_factor * _q
        for ind, _done in enumerate(done):
            if _done:
                q_target[ind] = -1

        loss = tf.reduce_sum(
            tf.square(q_target -
                      tf.reduce_max(q(previous_observations / 255.), axis=1)))
    grads = tape.gradient(loss, q.trainable_variables)
    optimizer.apply_gradients(zip(grads, q.trainable_variables))

    return loss


steps = 0

for i_episode in range(5000):

    print('Episode : {}, current replay memory size {}'.format(
        i_episode + 1, memory.counter))

    if update_target_counter > 0 and update_target_counter == update_interval:
        q_hat.set_weights(q.get_weights())
        update_target_counter = 0

    observation = env.reset()

    quad_obs = []
    observation1 = np.array([])
    observation2 = np.array([])

    action = 0
    for t in range(1, 501):
        quad_obs.append(observation)

        if t % 4 == 0:
            action = select_a_with_epsilon_greedy(quad_obs, eps)

        observation, reward, done, info = env.step(action)
        if len(quad_obs) == 4:
            inp = preprocess_input(quad_obs)
            if observation1.size == 0:
                observation1 = inp
            else:
                observation2 = inp
            quad_obs = []

        if observation1.size != 0 and observation2.size != 0:
            experience = [None] * 5
            experience[0] = observation1  # previous observation
            experience[1] = action
            experience[2] = reward
            experience[3] = observation2  # next observation
            experience[4] = done  # store done to add negative reward on death
            memory.add_experience(experience)
            observation1 = np.array([])
            observation2 = np.array([])
        if memory.counter > 5000:
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
