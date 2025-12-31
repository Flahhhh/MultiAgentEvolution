import os
import datetime
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from pygad.pygad import GA
from pygad.torchga import torchga

from utils import ma_battle_fit, ma_battle_fit_random, ma_battle_fit_best
from net import MANetBase, MAFCNet, MAConvNet

archs = {
    "MAFCNet": MAFCNet,
    "MAConvNet": MAConvNet
}

def run():
    root_dir = f"logs/{str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))}"
    print(root_dir)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    model_dir = os.path.join(root_dir, "Models")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    CONFIG = {
        "name": "MANNE",
        "arch": "MAFCNet",
        "epochs": 100,
        "population_size": 256,
        "num_parents_mating": 4,
        "num_process": 8,
        "device": "cuda:0",
    }
    device = torch.device(CONFIG["device"])

    with open(os.path.join(root_dir, "config.json"), 'w') as json_file:
        json.dump(CONFIG, json_file, indent=4)

    print("[INFO]: Config data saved")

    arch_cls = archs[CONFIG["arch"]]
    model = arch_cls().to(device)
    writer = SummaryWriter(os.path.join(root_dir, f"learning-{CONFIG['name']}"), comment="-" + CONFIG["name"],
                           flush_secs=120)

    def callback_generation(ga_instance: GA):
        epoch = ga_instance.generations_completed
        best_fitness = ga_instance.best_solution()[1]
        mean_fitness = sum(ga_instance.best_solutions_fitness) / len(ga_instance.best_solutions_fitness)

        print("[INFO]: Generation = {generation}".format(generation=epoch))
        print("[INFO]: Best fitness = {fitness}".format(fitness=best_fitness))
        print("[INFO]: Mean fitness = {fitness}".format(fitness=mean_fitness))

        writer.add_scalar("data/best_fitness", best_fitness, epoch)
        writer.add_scalar("data/mean_fitness", mean_fitness, epoch)

        if ga_instance.generations_completed % 5 == 0:
            state = {'info': "JanggiBotV1",  # описание
                     'date': datetime.datetime.now(),  # дата и время
                     'epochs': epoch,
                     'model': torchga.model_weights_as_dict(model, ga_instance.best_solution()[0]),  # параметры модели
                     # 'optimizer': agent.model.optimizer.state_dict() # состояние оптимизатора
                     }
            str_dir = os.path.join(root_dir, f'Models/{CONFIG["name"]}-{epoch}.pt')
            torch.save(state, str_dir)

    torch_ga = torchga.TorchGA(model, CONFIG["population_size"])
    ga_instance = GA(num_generations=CONFIG["epochs"], num_parents_mating=CONFIG["num_parents_mating"],
                     fitness_func=ma_battle_fit_best, on_generation=callback_generation,
                     initial_population=torch_ga.population_weights, parallel_processing=CONFIG["num_process"], )
    ga_instance.run()
    ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("[INFO]: Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("[INFO]: Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

    model_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_dict)

    state = {'info': "JanggiBotV1",  # описание
             'date': datetime.datetime.now(),  # дата и время
             'epochs': CONFIG["epochs"],
             'model': model.state_dict(),  # параметры модели
             # 'optimizer': agent.model.optimizer.state_dict() # состояние оптимизатора
    }
    str_dir = os.path.join(root_dir, f'Models/{CONFIG["name"]}.pt')
    torch.save(state, str_dir)

if __name__=="__main__":
    run()
