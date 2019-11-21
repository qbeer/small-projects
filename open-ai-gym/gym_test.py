import gym

env = gym.make('Breakout-v0')


def do_action_for_t(action, T):
    env.reset()
    t = 0
    while t < T:
        env.render()
        action = action
        _, _, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        t += 1


do_action_for_t(0, 1000)  # nothing
#do_action_for_t(1, 1000)  # fire - get new ball if none is present
#do_action_for_t(2, 1000) # right
#do_action_for_t(3, 1000) # left

env.close()