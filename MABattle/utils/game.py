import torch
from torch import nn
import random

from MABattle.MABattleV0.env import MABattleEnv



def play_game(model: nn.Module, model_, env: MABattleEnv, device, zero_sum=True):
    agent_turn = random.choice((-1, 1))

    models = {agent_turn: model, -agent_turn: model_}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    while not finished:
        legals = info["legals"]

        actions = models[env.unwrapped.to_play()].get_actions(state.to(device), legals.to(device))
        c += 1
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        if env.unwrapped.to_play() == -agent_turn:
            rewards += reward
        elif zero_sum:
            rewards -= reward

    return rewards


def test_play_game(model: nn.Module, model_, env: MABattleEnv, device):
    agent_turn = 1

    models = {agent_turn: model, -agent_turn: model_}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    images = []
    while not finished:
        legals = info["legals"]

        actions = models[env.unwrapped.to_play()].get_actions(state.to(device), legals.to(device))
        c += 1
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        images.append(env.unwrapped.get_image())

        if env.unwrapped.to_play() == -agent_turn: rewards += reward

    images[0].save(f"output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    return rewards
