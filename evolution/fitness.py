import random

from .evolution__ import Evolution
from MABattle.utils.env import make_env

import gymnasium as gym

from MABattle.utils.game import play_game
from net import MAFCNet
from const import device, env_name, num_games
from utils.agent import RandomAgent

cls = MAFCNet


def ma_battle_fit(model, evolution: Evolution):

    env = make_env()
    model = model.to(device)

    fit = 0.

    for _ in range(num_games):
        model_ = random.choice(evolution.population).to(device)
        fit += play_game(model, model_, env, device)

    return fit / num_games

def ma_battle_fit_random(model, evolution: Evolution):

    env = make_env()
    model, model_ = model.to(device), RandomAgent()

    fit = 0.

    for _ in range(num_games):
        fit += play_game(model, model_, env, device)

    return fit / num_games

random_percent=0.25
def ma_battle_fit_best(model, best):

    env = make_env()
    model, model_ = model.to(device), best

    random_agent = RandomAgent()
    fit = 0.
    opponents = [random_agent] * int(num_games * random_percent) + [model_] * (num_games - int(num_games * random_percent))

    for i in range(num_games):
        fit += play_game(model, opponents[i], env, device)

    return fit / num_games
