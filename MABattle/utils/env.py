import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch

from const import device, env_name, num_games

def make_env():
    env = NumpyToTorch(gym.make(env_name))

    return env
