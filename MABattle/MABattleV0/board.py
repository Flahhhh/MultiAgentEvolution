import torch

from .const import BOARD_SHAPE, BOARD_SIZE, NUM_LINES, ROW_RANGE, COL_RANGE, UNIT_POSSIBLE_ACTIONS, \
    CAPTURE_REWARD, WIN_REWARD
from .units import Unit

import numpy as np


class Board:
    units: dict[int: dict[tuple[int, int]: Unit]]
    turn: int

    def __init__(self, use_flip=True):
        self.reset()

        self.use_flip = use_flip
        self.units = {1: {}, -1: {}}

    def reset(self):
        self.turn = 1
        self._spawn_units()
        # self._make_board()

    def flip(self):
        # self.board = np.flip(self.board)
        for p, units in self.units.items():
            units_ = {}
            for pos, unit in units.items():
                new_pos = (BOARD_SHAPE[0] - pos[0] - 1, BOARD_SHAPE[1] - pos[1] - 1)
                unit.pos = new_pos
                units_[new_pos] = unit
            self.units[p] = units_

    def _make_board(self):
        ...
        # self.board = np.empty(BOARD_SHAPE)

    def _spawn_units(self):
        self.units = {1: {}, -1: {}}
        line_idxs = list(range(BOARD_SHAPE[0]))
        row_idxs = list(range(BOARD_SHAPE[1]))

        for i in range(NUM_LINES):
            for j in range(BOARD_SHAPE[1]):
                pos = (line_idxs[i], j)
                idx = i * BOARD_SIZE[1] + j + 1
                #idx = (i, j)
                unit = Unit(pos, 1, idx)
                self.units[1][pos] = unit
                # self.board[*pos] = unit

        for i in range(NUM_LINES):
            for j in range(BOARD_SHAPE[1]):
                pos = (line_idxs[::-1][i], row_idxs[::-1][j])
                idx = i * BOARD_SIZE[1] + j + 1
                #idx = (i, j)
                unit = Unit(pos, -1, idx)
                self.units[-1][pos] = unit

    def _validate_move(self, move: np.ndarray, unit: Unit) -> bool:
        #print(self.units, "SSSS")
        #print(unit.pos.shape, move.shape, "SSSSSS")
        new_pos = unit.pos + move

        return new_pos[0] in ROW_RANGE and new_pos[1] in COL_RANGE

    def get_possible_actions(self) -> dict[int: list[int]]:
        moves = {}
        for _, unit in self.units[self.turn].items():
            moves[unit.idx] = []
            for move_idx in range(len(UNIT_POSSIBLE_ACTIONS)):
                if self._validate_move(UNIT_POSSIBLE_ACTIONS[move_idx], unit):
                    moves[unit.idx].append(move_idx)

        return moves

    def get_alive(self):
        r = []
        for _, unit in self.units[self.turn].items():
            #print(self.units, self.units[self.turn], "SSSSS")
            r.append(unit.idx)

        return r



    def _apply_action(self, action: int, unit: Unit) -> int:
        pos = unit.pos
        new_pos = unit.pos + UNIT_POSSIBLE_ACTIONS[action]

        cur_units = self.units[self.turn]
        opp_units = self.units[-self.turn]

        #print(tuple(new_pos), tuple(new_pos).__class__)
        #print(next(iter(self.units[1].keys())).__class__, "ZZZ")
        if action == 0 or cur_units.get(tuple(new_pos), None) is not None:
            return 0

        r = 0
        if opp_units.get(tuple(new_pos), None) is not None:
            opp_units.__delitem__(tuple(new_pos))
            r = CAPTURE_REWARD

        #print(self.units[self.turn], pos)
        cur_units.__delitem__(tuple(pos))
        cur_units[tuple(new_pos)] = unit
        unit.pos = new_pos

        return r

    def _get_done(self):
        return len(self.units[-self.turn]) == 0

    def step(self, actions: dict[int: int]):
        #print(actions)
        possible_move_sets = self.get_possible_actions()
        units = self.units[self.turn]
        r = []

        for pos in list(units.keys()):
            unit = units[pos]
            idx = unit.idx
            action = actions[idx]
            if action not in possible_move_sets[idx]:
                raise Exception(f"MOVE IDX: {action} | UNIT IDX: {unit.idx} | NEW POS: {unit.pos+UNIT_POSSIBLE_ACTIONS[action]} | LEGALS: {possible_move_sets[idx]} | VALIDATION: {action in possible_move_sets[idx]} | UNITS: {self.units[1]} | [ERROR]: Invalid move({idx, action})")

            r.append(self._apply_action(action, unit))

        d = self._get_done()
        #if d:
        #    r.append(WIN_REWARD)

        self.turn *= -1
        if self.use_flip:
            self.flip()

        return r, d
