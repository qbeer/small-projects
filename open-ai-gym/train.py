import tensorflow as tf
import datetime as dt

train_writer = tf.summary.create_file_writer(
    "./logs" + f"/DQN_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

import gym
import numpy as np
from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork

env = gym.make('Breakout-v0')

n_actions = env.action_space.n
discount_factor = 0.95
update_interval = 1000
eps = 0.4

# actions are : NOPE, FIRE (new ball), RIGHT, LEFT
q = DeepQNetwork(n_actions)
q_hat = DeepQNetwork(n_actions)

# Initialize both networks with the same weights
q_hat.set_weights(q.get_weights())

memory = ReplayMemory(10_000)

optimizer = tf.keras.optimizers.RMSprop()

q.compile(optimizer=optimizer, loss='mse')


# epsilon - greedy algorithm
def select_a_with_epsilon_greedy(state, epsilon=eps):
    q_value = q(np.expand_dims(state / 255., axis=0)).numpy()
    action = np.argmax(q_value)
    if np.random.rand() < epsilon:
        action = np.random.randint(4)
    return action


update_target_counter = 0


@tf.function
def train_decision_network(counter):
    if counter > 32 * 3:
        experience_samples = memory.sample_experiences(32)
        previous_observations = np.array(
            [sample[0] for sample in experience_samples])
        updated_observations = np.array(
            [sample[3] for sample in experience_samples])
        rewards = np.array([sample[1] for sample in experience_samples])

        with tf.GradientTape() as tape:
            q_target = rewards + discount_factor * q_hat(
                updated_observations / 255.)
            loss = q.train_on_batch(
                tf.convert_to_tensor(previous_observations / 255.),
                tf.convert_to_tensor(q_target))

        if update_target_counter < update_interval:
            update_target_counter += 1

        return loss
    return 0


for i_episode in range(1_000):

    print('Episode : {}'.format(i_episode + 1))

    if update_target_counter == update_interval:
        q_hat.set_weights(q.get_weights())
        update_target_counter = 0

    observation = env.reset()
    for t in range(100):
        experience = [None] * 4
        experience[0] = observation  # previous observation
        env.render()
        action = select_a_with_epsilon_greedy(observation)
        observation, reward, done, info = env.step(action)
        experience[1] = action
        experience[2] = reward
        experience[3] = observation  # next observation
        memory.add_experience(experience)
        loss = train_decision_network(memory.counter)

        print('Loss : {}, Counter : {}'.format(loss, memory.counter))

        if done:
            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=(i_episode + 1) * t)
            break

env.close()

for obs, act, r, _obs in memory.experiences:
    if r > 0:
        print(obs.shape, act, r, _obs.shape)