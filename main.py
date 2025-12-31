import os
import datetime
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from pygad.pygad import GA
from pygad.torchga import torchga

from utils import ma_battle_fit, ma_battle_fit_random, ma_battle_fit_best
from net import MANetBase, MAFCNet, MAConvNet
from evolution import Evolution

archs = {
    "MAFCNet": MAFCNet,
    "MAConvNet": MAConvNet
}

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evolve = Evolution(model_cls=MAFCNet, fitness_fn=ma_battle_fit_best, population_size=128, device=device)

    evolve.train(100)

if __name__=="__main__":
    run()
