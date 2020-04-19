import numpy as np
import cv2

HEIGHT = 84
WIDTH = 84

N_STACK = 4


def preprocess_input(observations):
    observed = []
    for obs in observations:
        # to gray and to 0-1
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #print(obs.shape, np.min(obs), np.max(obs))
        cv2.imwrite('grey.png', obs)
        obs = cv2.resize(obs, (WIDTH, HEIGHT))
        #cv2.imwrite('resized.png', obs)
        observed.append(obs)
    observed = np.array(observed)
    observed = np.transpose(observed, (1, 2, 0))
    return observed


import gym

env = gym.make('Breakout-v0')

obs = env.reset()

all_obs = [obs]

for action in np.random.randint(0, 4, size=3):
    obs, _, _, _ = env.step(1) # fire a ball
    all_obs.append(obs)

preprocessed = preprocess_input(all_obs)

for ind in range(N_STACK):
    print(preprocessed.shape)
    cv2.imwrite("%d.png" % ind, preprocessed[..., ind])
