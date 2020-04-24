import numpy as np
import cv2
import gym

IMG_HEIGHT = 64
IMG_WIDTH = 48
STACK_SIZE = 4

env = gym.make('BreakoutDeterministic-v4')

for D in range(10):
    env.reset()

    frames = [env.step(env.action_space.sample())[0] for _ in range(STACK_SIZE)]

    def preprocess_input(frames):
        observed = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, STACK_SIZE))
        for ind, obs in enumerate(frames):
            # to gray and to 0-1
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (IMG_WIDTH, IMG_HEIGHT))
            observed[..., ind] += obs
        return observed.astype(np.float32)

    state = preprocess_input(frames)

    [cv2.imwrite('%d-%d.png' % (D, i), state[..., i]) for i in range(STACK_SIZE)]