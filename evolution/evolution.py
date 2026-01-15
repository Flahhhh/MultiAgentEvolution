from copy import deepcopy
from multiprocessing import Pool, Process

import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
from utils import RandomAgent


class Evolution:
    def __init__(self, model_cls, fitness_fn, population_size, device, callback_fn=None):
        self.num_processes = 8

        self.population_size = population_size
        self.model_cls = model_cls
        self.device = device
        self.fitness_fn = fitness_fn
        self.callback_fn = callback_fn

        self.population = self.generate_initial_population()
        self.fitness = torch.FloatTensor([0] * self.population_size)

        self.epoch = 0
        self.elite_amount = 16
        self.num_relatives = 4
        self.init_noise_scale = 0.35
        self.noise_scale = self.init_noise_scale
        self.min_noise_scale = 0.005
        self.mutation_amount = self.population_size - self.elite_amount
        self.best = RandomAgent()

    def _update_noise_scale(self, epoch, epochs):
        self.noise_scale = self.init_noise_scale - (self.init_noise_scale - self.min_noise_scale) * epoch / epochs

    def generate_initial_population(self):
        return np.array([self.model_cls().to(self.device) for _ in range(self.population_size)])

    def fit(self):
        with Pool(self.num_processes) as pool:
            self.fitness = torch.tensor(pool.starmap(self.fitness_fn, [(model, self.best) for model in self.population]))

        self.best = self.population[self.fitness.argmax()]

    def get_best(self):
        return self.best

    def generate_new_population(self):
        new_population = np.array([None] * self.population_size)

        elite_ids = torch.topk(self.fitness, self.elite_amount).indices

        new_population[:self.elite_amount] = self.population[elite_ids]

        for idx in range(self.elite_amount, self.population_size):
            solution = deepcopy(self.population[idx])
            relatives_idx = self._select_relatives_idx()
            relatives = self.population[relatives_idx]

            relatives_fitness = self.fitness[relatives_idx]
            weights = softmax(relatives_fitness, dim=0).to(self.device)

            with torch.no_grad():
                for i, p in enumerate(solution.parameters()):
                    params_stack = torch.stack([list(r.parameters())[i] for r in relatives])
                    weights_expanded = weights.view(-1, *([1] * (params_stack.dim() - 1)))

                    p.copy_(
                        torch.sum(weights_expanded*params_stack, dim=0) +
                        torch.randn_like(p) * self.noise_scale
                    )

            new_population[idx] = solution

        self.population = new_population
        return new_population

    def _get_probs(self):
        return softmax(self.fitness, dim=0)

    def _select_relatives_idx(self):
        return torch.multinomial(self._get_probs(), self.num_relatives, replacement=False)

    def train(self, epochs):
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            self.epoch = epoch
            self.fit()
            self.generate_new_population()

            self.best_fitness, self.mean_fitness = float(self.fitness.max()), float(self.fitness.mean())
            self._update_noise_scale(epoch, epochs)
            if self.callback_fn is not None:
                self.callback_fn(self)

            pbar.set_postfix({"BEST": self.best_fitness, "MEAN": self.mean_fitness})
