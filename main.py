import os
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from const import device as device_name
from net import MAFCNet, MAConvNet
from evolution import Evolution, ma_battle_fit, ma_battle_fit_random, ma_battle_fit_best

archs = {
    "MAFCNet": MAFCNet,
    "MAConvNet": MAConvNet
}

def run():
    name = "nne"
    root_dir = f"logs/{str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))}"
    print(root_dir)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    model_dir = os.path.join(root_dir, "Models")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    writer = SummaryWriter(os.path.join(root_dir, f"learning-{name}"), comment="-" + name,
                           flush_secs=120)

    def callback(evolution: Evolution):
        epoch = evolution.epoch

        writer.add_scalar("data/best_fitness", evolution.best_fitness, epoch)
        writer.add_scalar("data/mean_fitness", evolution.mean_fitness, epoch)
        writer.add_scalar("data/noise_scale", evolution.noise_scale, epoch)

        if epoch % 5 == 0:
            state = {'info': "NNE-V1",  # описание
                     'date': datetime.datetime.now(),  # дата и время
                     'epochs': epoch,
                     'model': evolution.best.state_dict(),
                     }
            str_dir = os.path.join(root_dir, f'Models/{name}-{epoch}.pt')
            torch.save(state, str_dir)

    device = torch.device(device_name)
    evolve = Evolution(
        model_cls=MAFCNet,
        fitness_fn=ma_battle_fit_random,
        population_size=128,
        device=device,
        callback_fn=callback
    )

    evolve.train(150)

    state = {'info': "NNE-V1",  # описание
             'date': datetime.datetime.now(),  # дата и время
             'epochs': evolve.epoch,
             'model': evolve.best.state_dict(),
             }
    str_dir = os.path.join(root_dir, f'Models/{name}.pt')
    torch.save(state, str_dir)
if __name__=="__main__":
    run()
