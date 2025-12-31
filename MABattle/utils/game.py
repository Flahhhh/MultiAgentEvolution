import torch
from torch import nn
import random

from MABattle.MABattleV0.env import MABattleEnv



def play_game(model: nn.Module, model_, env: MABattleEnv, device, zero_sum=True):
    agent_turn = random.choice((-1, 1))
    # print(env, "SSS")

    models = {agent_turn: model, -agent_turn: model_}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    while not finished:
        # print(env, "SSS")
        # state, reward, finished, _, _ = env.step(models[env.unwrapped.to_play()](state.to(device)))

        # print(state.unsqueeze(0).shape, state)
        legals = info["legals"]

        actions = models[env.unwrapped.to_play()].get_actions(state.to(device), legals.to(device))
        #actions = models[env.unwrapped.to_play()].get_actions(state.to(device).unsqueeze(0), legals.to(device))
        c += 1
        #print(c, "SSSSSS")
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        #env.render().save(f"{c}.png")

        if env.unwrapped.to_play() == -agent_turn:
            rewards += reward
        elif zero_sum:
            rewards -= reward

    return rewards


def test_play_game(model: nn.Module, model_, env: MABattleEnv, device):
    agent_turn = 1
    #agent_turn = random.choice((-1, 1))
    # print(env, "SSS")

    models = {agent_turn: model, -agent_turn: model_}

    state, info = env.reset()
    finished = False
    rewards = 0.0
    c = 0
    images = []
    while not finished:
        # print(env, "SSS")
        # state, reward, finished, _, _ = env.step(models[env.unwrapped.to_play()](state.to(device)))

        # print(state.unsqueeze(0).shape, state)
        legals = info["legals"]

        actions = models[env.unwrapped.to_play()].get_actions(state.to(device), legals.to(device))
        #actions = models[env.unwrapped.to_play()].get_actions(state.to(device).unsqueeze(0), legals.to(device))
        c += 1
        #print(c, "SSSSSS")
        state, reward, finished, truncated, info = env.step(actions)
        finished = finished or truncated

        images.append(env.render())

        if env.unwrapped.to_play() == -agent_turn: rewards += reward

    images[0].save(f"output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)
    return rewards
