import random
from pygad.pygad import GA
from pygad import torchga

import gymnasium as gym

from MABattle.utils.game import play_game
from net import MANetBase, MAFCNet
from const import device, env_name, num_games
from utils.agent import RandomAgent

cls = MAFCNet


def ma_battle_fit(ga_instance: GA, solution, sol_idx):
    #device = torch.device(device_name)

    env = gym.make(env_name)
    model, model_ = cls().to(device), cls().to(device)

    model_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_dict)

    fit = 0.

    for _ in range(num_games):
        opponent_solution = random.choice(ga_instance.population)
        model_dict_ = torchga.model_weights_as_dict(model=model_, weights_vector=opponent_solution)
        model_.load_state_dict(model_dict_)

        fit += play_game(model, model_, env, device)

    return fit / num_games

def ma_battle_fit_random(ga_instance: GA, solution, sol_idx):
    #device = torch.device(device_name)

    env = gym.make(env_name)
    model, model_ = cls().to(device), RandomAgent()

    model_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_dict)

    fit = 0.

    for _ in range(num_games):
        fit += play_game(model, model_, env, device)

    return fit / num_games

random_percent=0.25
def ma_battle_fit_best(ga_instance: GA, solution, sol_idx):
    # device = torch.device(device_name)

    env = gym.make(env_name)
    model, model_ = cls().to(device), cls().to(device)

    model_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_dict)

    if len(ga_instance.best_solutions)==0:
        solution_ = random.choice(ga_instance.population)
    else:
        solution_ = ga_instance.best_solutions[0]

    model_dict_ = torchga.model_weights_as_dict(model=model_, weights_vector=solution_)
    model_.load_state_dict(model_dict_)

    random_agent = RandomAgent()
    fit = 0.
    opponents = [random_agent] * int(num_games * random_percent) + [model_] * (num_games - int(num_games * random_percent))

    for i in range(num_games):
        fit += play_game(model, opponents[i], env, device)

    return fit / num_games