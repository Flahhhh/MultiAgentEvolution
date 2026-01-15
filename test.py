import torch
import gymnasium as gym

from MABattle.utils import test_play_game
from MABattle.utils.env import make_env
from const import env_name, device
from utils import RandomAgent
from net import MAFCNet

model = MAFCNet().to(device).eval()
state_dict = torch.load(r"C:\Users\YOU-LA\Desktop\apple\genetic_algorithms\nne\logs\2026-01-08 19-38\Models\nne.pt",
                        weights_only=False)["model"]
model.load_state_dict(state_dict)

model_ = RandomAgent()
env = make_env()  # gym.make(env_name, render_mode="human")

print(test_play_game(model, model_, env, device))
