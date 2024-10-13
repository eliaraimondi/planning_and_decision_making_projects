from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc, Cell
from pdm4ar.exercises_def.ex05.comparison import start_end_configurations_are_equal


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""

        north = (state[0] - 1, state[1])
        west = (state[0], state[1] - 1)
        south = (state[0] + 1, state[1])
        east = (state[0], state[1] + 1)
        adj_compl = [north, west, south, east]
        adj = [(x, y) for x, y in adj_compl if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]]
        val_adj = [self.grid[vert] for vert in adj]
        num_wormhole = np.count_nonzero(self.grid == Cell.WORMHOLE)
        prob = 0

        if self.grid[next_state] == Cell.WORMHOLE:
            prob_wormhole = 1 / num_wormhole
        else:
            prob_wormhole = 1
        num_wormhole_adj_dir = 0
        num_wormhole_adj = 1
        num_oth_wormhole = 0

        match self.grid[state]:

            case Cell.START | Cell.GRASS | Cell.WORMHOLE:
                match action:
                    case Action.NORTH | Action.WEST | Action.SOUTH | Action.EAST:
                        if self.grid[next_state] == 4:
                            num_wormhole_adj_dir = len([i for i in adj if self.grid[i] == 4 and i == adj_compl[action]])
                            num_wormhole_adj = len([i for i in adj if self.grid[i] == 4 and i != adj_compl[action]])
                            num_oth_wormhole = num_wormhole_adj
                        if (
                            adj_compl[action] not in adj
                            or self.grid[adj_compl[action]] == 5
                            or self.grid[next_state] == 5
                            or next_state == state
                        ):
                            return 0
                        elif next_state == adj_compl[action]:
                            prob = (0.75 + 0.25 / 3 * num_oth_wormhole) * prob_wormhole
                        elif next_state in [x for x in adj_compl if x != adj_compl[action]]:
                            prob = (0.25 / 3 * num_wormhole_adj + 0.75 * num_wormhole_adj_dir) * prob_wormhole
                        elif self.grid[next_state] == Cell.WORMHOLE:
                            prob = (0.25 / 3 * num_wormhole_adj + 0.75 * num_wormhole_adj_dir) * prob_wormhole
                        if self.grid[next_state] == 1:
                            adj_cliff = val_adj.count(5) + len([i for i in adj_compl if i not in adj])
                            prob += 0.25 / 3 * adj_cliff

                    case Action.ABANDON:
                        if self.grid[next_state] == 1:
                            prob = 1

            case Cell.SWAMP:
                match action:
                    case Action.NORTH | Action.WEST | Action.SOUTH | Action.EAST:
                        if self.grid[next_state] == 4:
                            num_wormhole_adj_dir = len([i for i in adj if self.grid[i] == 4 and i == adj_compl[action]])
                            num_wormhole_adj = len([i for i in adj if self.grid[i] == 4 and i != adj_compl[action]])
                            num_oth_wormhole = num_wormhole_adj
                        if (
                            adj_compl[action] not in adj
                            or self.grid[adj_compl[action]] == 5
                            or self.grid[next_state] == 5
                        ):
                            return 0
                        elif next_state == adj_compl[action]:
                            prob = (0.5 + 0.25 / 3 * num_oth_wormhole) * prob_wormhole
                        elif next_state in [x for x in adj_compl if x != adj_compl[action]]:
                            prob = (0.25 / 3 * num_wormhole_adj + 0.5 * num_wormhole_adj_dir) * prob_wormhole
                        elif next_state == state:
                            prob = 0.2
                        elif self.grid[next_state] == Cell.WORMHOLE:
                            prob = (0.25 / 3 * num_wormhole_adj + 0.5 * num_wormhole_adj_dir) * prob_wormhole
                        if self.grid[next_state] == 1:
                            adj_cliff = val_adj.count(5) + len([i for i in adj_compl if i not in adj])
                            prob += 0.25 / 3 * adj_cliff + 0.05

                    case Action.ABANDON:
                        if self.grid[next_state] == 1:
                            prob = 1

            case Cell.CLIFF:
                if self.grid[next_state] == 1:
                    prob = 1

            case Cell.GOAL:
                if action == Action.STAY:
                    prob = 1

        return prob

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:

        develop_new_robot = -10
        hour = -1
        compleate = 50

        match self.grid[state]:

            case Cell.START | Cell.GRASS | Cell.WORMHOLE:
                if action == Action.ABANDON:
                    reward = develop_new_robot
                elif self.grid[next_state] == 1 and (
                    (np.abs(next_state[0] - state[0]) + np.abs(next_state[1] - state[1])) > 1
                ):
                    reward = hour + develop_new_robot
                else:
                    reward = hour

            case Cell.SWAMP:
                if action == Action.ABANDON:
                    reward = develop_new_robot
                elif self.grid[next_state] == 1 and (
                    (np.abs(next_state[0] - state[0]) + np.abs(next_state[1] - state[1])) > 1
                ):
                    reward = 2 * hour + develop_new_robot
                else:
                    reward = 2 * hour

            case Cell.CLIFF:
                reward = develop_new_robot

            case Cell.GOAL:
                reward = compleate

        return reward


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass
