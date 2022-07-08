from floorplan_v0 import *


def test_env():
    env = FloorPlanEnv("./data")
    env.reset()
    env.render()

    # prev_screen = env.render(mode='rgb_array')
    # fg = figure()
    # ax = fg.gca()
    # visualizer = ax.imshow(prev_screen)

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # if i%5 == 0:
        # screen = env.render(mode='rgb_array')
        # visualizer.set_data(screen)
        # draw(), pause(1e-3)
        # Other option, just use: env.render() # mode=human by default
        env.render()
        if done:
            break
    env.close()


def testing():
    config = {
        "data_dir": "./data",
        "window_size": 1024
    }

    env = FloorPlanEnv(config)
    env.reset()
    env.render()

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    env.close()


def main():
    testing()


if __name__ == '__main__':
    main()


"""
# Using stable_baselines from openai
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

env = CustomEnv('/data')

# Training
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

# Testing
n_steps = 2000  # maximum number of steps per episode
episode_return = 0
obs = env.reset()
for i in range(n_steps):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  if done:
    break
  env.render()
  episode_return += rewards
print("Episode return:", episode_return)
"""
