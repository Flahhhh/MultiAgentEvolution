import torch
from torch import nn
from torchrl.modules.distributions import MaskedCategorical
from torch.nn.functional import relu, softmax

from MABattle.MABattleV0 import NUM_AGENTS, ACTION_SPACE
from const import device


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same", bias=True)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 10, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x, x.shape, x.dtype)
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv3(x)))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Logic2Net(nn.Module):
    def __init__(self):
        super(Logic2Net, self).__init__()

        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))  # softmax(self.fc2(x))

        return x

class MANetBase(nn.Module):
    def __init__(self):
        super(MANetBase, self).__init__()
        self.agents = []

    def forward(self, x):
        raise NotImplementedError()

    def get_actions(self, x, legals):
        actions = torch.full([NUM_AGENTS], 0, device=device)

        #print(f"INPUT: {x}")
        #x = self.forward(x).squeeze(0)
        #x = x.reshape(NUM_AGENTS, ACTION_SPACE)
        x = self.forward(x)
        #print(x.shape)
        #softmax(x, 1) * legals
        #print(f"MIDDLE: {x} | LEGALS: {legals}")
        mask = legals.any(1)


        dist = MaskedCategorical(logits=x[mask], mask=legals[mask])
        sample = dist.sample()

        actions[mask] = sample
        #print(x, x.shape, sample, sample.shape, )


        #print(legals, x)

        return actions

class MAConvNet(MANetBase):
    def __init__(self):
        super(MAConvNet, self).__init__()
        #self.max_pool = nn.MaxPool2d(2, 2)
        # self.batch_norm = nn.BatchNorm2d()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding="same", bias=True)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding="same", bias=True)
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, ACTION_SPACE * NUM_AGENTS)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x = self.max_pool(relu(self.conv1(x)))
        # x = self.max_pool(relu(self.conv2(x)))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)

        x = self.relu(self.fc1(x))  # relu(self.fc(x))
        x = self.fc2(x)
        # print(x.shape)

        return x


class MAFCNet(MANetBase):
    def __init__(self):
        super(MAFCNet, self).__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, ACTION_SPACE)
    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x