from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq  # you may find this helpful

from numpy import arcsin
from osmnx.distance import great_circle_vec
from zmq import NULL

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass


@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        queque = [start]
        costToReach = {}
        for node in list(self.graph.adj_list.keys()):
            costToReach[node] = 10000000
        costToReach[start] = 0

        parent = {start: NULL}

        while queque:
            min_val = 1000000000
            for node in queque:
                if costToReach[node] < min_val:
                    min_val = costToReach[node]
                    s = node

            if s == goal:
                return self.compute_path(s, parent, start)

            for next_node in self.graph.adj_list[s]:
                if self.graph.get_weight(s, next_node) is not None:
                    newCostToReach = costToReach[s] + self.graph.get_weight(s, next_node)
                if newCostToReach < costToReach[next_node]:
                    costToReach[next_node] = newCostToReach
                    parent[next_node] = s  # type: ignore
                    if next_node not in queque:
                        queque = queque + [next_node]

            queque.remove(s)  # type: ignore

    def compute_path(self, last_node, pre, start):
        ins = last_node
        path = []
        while ins != start:
            path = [ins] + path
            ins = pre[ins]
        return [start] + path


@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        coord_u = self.graph.get_node_coordinates(u)
        coord_v = self.graph.get_node_coordinates(v)

        distance = great_circle_vec(coord_u[0], coord_u[1], coord_v[0], coord_v[1], earth_radius=6371009)

        if distance < 600:
            speed = TravelSpeed.SECONDARY.value
        else:
            speed = TravelSpeed.HIGHWAY.value

        time_to_travel = distance / speed

        return time_to_travel

    def path(self, start: X, goal: X) -> Path:
        queque = [start]
        heapq.heapify(queque)
        costToReach = {}
        for node in list(self.graph.adj_list.keys()):
            costToReach[node] = 10000000
        costToReach[start] = 0
        distances = {start: self.heuristic(start, goal)}

        parent = {start: NULL}

        while queque:
            min_val = 1000000000
            for node in queque:
                if (costToReach[node] + distances[node]) < min_val:
                    min_val = costToReach[node] + distances[node]
                    s = node

            if s == goal:
                return self.compute_path(s, parent, start)

            for next_node in self.graph.adj_list[s]:
                weight = self.graph.get_weight(s, next_node)
                if weight is not None:
                    newCostToReach = costToReach[s] + weight
                if newCostToReach < costToReach[next_node]:
                    costToReach[next_node] = newCostToReach
                    parent[next_node] = s  # type: ignore
                    if next_node not in queque:
                        queque = queque + [next_node]
                        if next_node not in distances:
                            distances[next_node] = self.heuristic(next_node, goal)

            queque.remove(s)  # type: ignore

        return []

    def compute_path(self, last_node, pre, start):
        ins = last_node
        path = []
        while ins != start:
            path = [ins] + path
            ins = pre[ins]
        return [start] + path


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc  # type: ignore
    return total
