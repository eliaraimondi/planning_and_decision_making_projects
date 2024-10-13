from argparse import Action
from ast import While
import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Action
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        # matrix initialization
        new_value_func = np.zeros_like(grid_mdp.grid).astype(float)
        value_func = np.ones_like(grid_mdp.grid)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # moves and coordinates definition
        next_moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        actions = range(6)
        start = [tuple(coord) for coord in np.argwhere(grid_mdp.grid == 1)]
        coordinate = [coord for coord in list(np.ndindex(grid_mdp.grid.shape)) if grid_mdp.grid[coord] != 5]

        # find every wormhole
        wormholes = [tuple(coord) for coord in np.argwhere(grid_mdp.grid == 4)]

        probs = {}
        stage_rewards = {}
        next_states = {}

        for state in coordinate:
            # define next_states

            if grid_mdp.grid[state] == 0:
                next_states[state] = [state]
            else:
                next_states_compl = [(x + state[0], y + state[1]) for (x, y) in next_moves]
                if start[0] not in next_states_compl:
                    next_states_compl += start
                next_states[state] = [
                    (x, y)
                    for x, y in next_states_compl
                    if 0 <= x < grid_mdp.grid.shape[1] and 0 <= y < grid_mdp.grid.shape[0]
                ]
                # control if there are wormhole in the adjacent states, if yes add all the map's wormholes in next_states
                if any(grid_mdp.grid[x, y] == 4 for x, y in next_states[state]):
                    next_states[state] += [wormhole for wormhole in wormholes if wormhole not in next_states[state]]
                next_states[state] += [state]

            # precompute all the probs and rewards
            probs[state] = {}
            stage_rewards[state] = {}

            for action in actions:
                probs[state][action] = []
                stage_rewards[state][action] = []
                for i, j in next_states[state]:
                    probs[state][action].append(grid_mdp.get_transition_prob(state, Action(action), (i, j)))  # type: ignore
                    stage_rewards[state][action].append(grid_mdp.stage_reward(state, Action(action), (i, j)))  # type: ignore

        while True:
            delta = 0
            value_func = np.copy(new_value_func)

            for state in coordinate:
                value_func_vect = np.zeros(6)  #######
                for action in actions:
                    if np.abs(sum(probs[state][action]) - 1) > 0.0001:
                        value_func_vect[action] = -np.inf
                    else:
                        value_func_vect[action] = sum(
                            probs[state][action][ind]
                            * (stage_rewards[state][action][ind] + grid_mdp.gamma * value_func[next_states[state][ind]])
                            for ind in range(len(next_states[state]))
                        )

                new_value_func[state] = np.max(value_func_vect)
                policy[state] = np.argmax(value_func_vect)
                delta = max(delta, np.abs(new_value_func[state] - value_func[state]))

            if delta < 0.01:
                break

        return new_value_func, policy
