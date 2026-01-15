import random
#from pygad.pygad import GA
#from pygad import torchga

from .evolution__ import Evolution
from MABattle.utils.env import make_env

import gymnasium as gym

from MABattle.utils.game import play_game
from net import MANetBase, MAFCNet
from const import device, env_name, num_games
from utils.agent import RandomAgent

cls = MAFCNet


def ma_battle_fit(model, evolution: Evolution):
    #device = torch.device(device_name)

    env = make_env() #gym.make(env_name)
    model = model.to(device)

    fit = 0.

    for _ in range(num_games):
        model_ = random.choice(evolution.population).to(device)
        fit += play_game(model, model_, env, device)

    return fit / num_games

def ma_battle_fit_random(model, evolution: Evolution):
    #device = torch.device(device_name)

    env = make_env() #gym.make(env_name)
    model, model_ = model.to(device), RandomAgent()

    fit = 0.

    for _ in range(num_games):
        fit += play_game(model, model_, env, device)

    return fit / num_games

random_percent=0.25
def ma_battle_fit_best(model, best):
    # device = torch.device(device_name)

    env = make_env() #gym.make(env_name)
    model, model_ = model.to(device), best#evolution.best.to(device)

    random_agent = RandomAgent()
    fit = 0.
    opponents = [random_agent] * int(num_games * random_percent) + [model_] * (num_games - int(num_games * random_percent))

    for i in range(num_games):
        fit += play_game(model, opponents[i], env, device)

    return fit / num_games