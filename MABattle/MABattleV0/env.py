from typing import List, Optional, Any
from PIL import Image
import torch
import numpy as np

import gymnasium as gym
from gymnasium.core import ObsType

from MABattle.MABattleV0 import Board, UNIT_POSSIBLE_ACTIONS, BOARD_SHAPE, NUM_LINES, ACTION_SPACE, OBS_HIGH, \
    BOARD_SIZE, NUM_AGENTS, UNIT_IMAGE_SIZE, RED_UNIT_IMAGE, BLUE_UNIT_IMAGE, CELL_IMAGE


class MABattleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.flatten_observations = True
        self.shape = (NUM_AGENTS, BOARD_SHAPE[0]*BOARD_SHAPE[1]) if self.flatten_observations else (NUM_AGENTS, *BOARD_SIZE)

        self.observation_space = gym.spaces.Box(
            low=-1,  # -NUM_PIECE_TYPE,
            high=OBS_HIGH,  # NUM_PIECE_TYPE,
            shape=self.shape,  # (NUM_ROWS, NUM_COLS),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_SPACE)
        self.render_mode = render_mode
        self._board = Board()

        self.reward_type = "sum"
        #self.subtract_lost = True
        self.max_c = 50
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None, ):  # -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._board.reset()
        self.c = 0

        state = self.get_state()
        legals = self.get_legals()

        return state, {"legals": legals}

    def step(self, actions) -> tuple:
        #print(actions)
        rewards, done = self._board.step({idx: actions[idx-1] for idx in self._board.get_alive()})
        if self.reward_type == "mean":
            reward = sum(rewards)/len(rewards)
        elif self.reward_type == "sum":
            reward = sum(rewards)
        elif self.reward_type == "raw":
            reward = rewards

        state = self.get_state()

        self.c += 1
        #print(self.c)
        truncated = self.c >= self.max_c
        info = {"legals": self.get_legals()}

        # finished = done or truncated

        return state, reward, done, truncated, info

    def get_legals(self):
        legals = torch.full((NUM_AGENTS, ACTION_SPACE), False, dtype=torch.bool)
        legal_actions = self._board.get_possible_actions()

        for idx, move_idxs in legal_actions.items():
            #print(move_idxs)
            for move_idx in move_idxs:
                legals[idx-1, move_idx] = True

        #print(legals, legals.argmax(1))
        return legals

    def render(self):
        return self.get_state()

    def get_image(self):
        if self._board.turn == -1:
            self._board.flip()

        image = Image.new("RGB", (BOARD_SIZE[0] * UNIT_IMAGE_SIZE[0], BOARD_SIZE[1] * UNIT_IMAGE_SIZE[1]), "black")
        blue_image = Image.open(BLUE_UNIT_IMAGE).convert("RGBA")
        red_image = Image.open(RED_UNIT_IMAGE).convert("RGBA")
        cell_image = Image.open(CELL_IMAGE).convert("RGBA")
        mask = red_image.split()[3]

        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                image.paste(cell_image, (j * UNIT_IMAGE_SIZE[0], i * UNIT_IMAGE_SIZE[1]))

                if (i, j) in self._board.units[1]:
                    image.paste(blue_image, (j * UNIT_IMAGE_SIZE[0], i * UNIT_IMAGE_SIZE[1]), mask=mask)
                    # image.putpixel((i,j), (255,0,0))
                elif (i, j) in self._board.units[-1]:
                    # image.putpixel((i,j), (0,0,255))
                    image.paste(red_image, (j * UNIT_IMAGE_SIZE[0], i * UNIT_IMAGE_SIZE[1]), mask=mask)

        if self._board.turn == -1:
            self._board.flip()

        return image

    def close(self):
        self._board = None
        return

    def to_play(self):
        return self._board.turn

    def get_state(self):
        state = np.zeros(shape=(NUM_AGENTS, *BOARD_SIZE), dtype=np.float32)
        #state = torch.zeros(NUM_AGENTS, *BOARD_SIZE)

        for pos, unit in self._board.units[self._board.turn].items():
            #print(pos, state[pos[0], pos[1]], state, "SSSS")
            # state[*pos] = unit.idx

            state[:, pos[0], pos[1]] = 1#unit.idx
            state[unit.idx-1, pos[0], pos[1]] = 2

        for pos, unit in self._board.units[-self._board.turn].items():
            # state[*pos] = -1
            state[:, pos[0], pos[1]] = -1

        #print(state.shape)
        # for i in range(BOARD_SHAPE[0]):
        #    for j in range(BOARD_SHAPE[1]):
        #        obj = self.board[i, j]
        #        if obj != None:
        #            rel_player = obj.player * self.turn
        #            state[i, j] = obj.idx if rel_player==1 else -1
        if self.flatten_observations:
            state = np.reshape(state, shape=[NUM_AGENTS, BOARD_SHAPE[0]*BOARD_SHAPE[1]])
            #state = torch.reshape(state, shape=[4, BOARD_SHAPE[0]*BOARD_SHAPE[1]])

        return state

    def __getstate__(self):
        return self.get_state()
