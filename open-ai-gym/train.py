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

EPS_ANNEALING_INTERVAL = 100000

sample_size = 128

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
        observed.append(cv2.resize(obs, (84, 60)))
    observed = np.array(observed)
    return observed.reshape((HEIGHT, WIDTH, len(observations)))


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

    done = np.array([sample[4] for sample in experience_samples])

    with tf.GradientTape() as tape:
        _q = tf.reduce_max(q_hat(updated_observations / 255.), axis=1)
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

    print('Episode : {}'.format(i_episode + 1))

    if update_target_counter > 0 and update_target_counter == update_interval:
        q_hat.set_weights(q.get_weights())
        update_target_counter = 0

    observation = env.reset()

    quad_obs = []
    observation1 = None
    observation2 = None
    for t in range(100):
        if t % 4 == 0:
            action = select_a_with_epsilon_greedy(observation, eps)

        observation, reward, done, info = env.step(action)
        quad_obs.append(observation)
        if len(quad_obs) == 4:
            inp = preprocess_input(quad_obs)
            if observation1 == None:
                observation1 = inp
            else:
                observation2 = inp

        if observation1 != None and observation2 != None:
            experience = [None] * 5
            experience[0] = observation1  # previous observation
            experience[1] = action
            experience[2] = reward
            experience[3] = observation2  # next observation
            experience[4] = done  # store done to add negative reward on death
            memory.add_experience(experience)
        if memory.counter > 25000:
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
