import gym

env = gym.make('SpaceInvaders-ram-v0',render_mode='human')
observeration = env.reset()

print(observeration)
print(env.action_space)

print(observeration)
print(env.action_space)

done = False
while not done:
    observeration, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()

env.close()
