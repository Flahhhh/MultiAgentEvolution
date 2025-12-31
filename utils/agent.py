import torch
import random

from MABattle.MABattleV0.const import NUM_AGENTS, ACTION_SPACE

class RandomAgent:
    def __init__(self):
        pass

    def get_actions(self, _, legals: torch.BoolTensor):
        actions = torch.full([NUM_AGENTS], 0)

        mask = legals.sum(1)
        for idx in range(NUM_AGENTS):
            if not mask[idx]: continue
            actions[idx] = random.choice(torch.argwhere(legals[idx]))

        return actions