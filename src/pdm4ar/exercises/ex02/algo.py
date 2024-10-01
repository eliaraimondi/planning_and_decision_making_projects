from abc import abstractmethod, ABC

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        # pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        queque = [start]
        opened_nodes = []
        pre = {}

        while len(queque) != 0:
            node = queque[0]
            opened_nodes.append(node)
            queque.pop(0)
            path = []
            if node == goal:
                ins = node
                while ins != start:
                    path = [ins] + path
                    ins = pre[ins]
                path = [start] + path
                return path, opened_nodes
            news = []
            for next_node in graph[node]:
                if next_node not in opened_nodes and next_node not in queque:
                    news.append(next_node)
                    pre[next_node] = node
            news.sort()
            queque = news + queque

        return [], opened_nodes


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        queque = [start]
        opened_nodes = []
        pre = {}

        while len(queque) != 0:
            node = queque[0]
            opened_nodes.append(node)
            queque.pop(0)
            path = []
            if node == goal:
                ins = node
                while ins != start:
                    path = [ins] + path
                    ins = pre[ins]
                path = [start] + path
                return path, opened_nodes
            news = []
            for next_node in graph[node]:
                if next_node not in opened_nodes and next_node not in queque:
                    news.append(next_node)
                    pre[next_node] = node
            news.sort()
            queque += news

        return [], opened_nodes


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        for d in range(len(graph) + 1):
            queque = [start]
            opened_nodes = []
            pre = {}
            max_depth = 0

            while len(queque) != 0 and max_depth < d:
                node = queque[0]
                opened_nodes.append(node)
                queque.pop(0)

                if node == goal:
                    return self.compute_path(node, pre, start), opened_nodes
                news = []

                for next_node in graph[node]:
                    old = -1
                    try:
                        old = pre[next_node]
                    except KeyError:
                        pass

                    pre[next_node] = node
                    if (
                        next_node not in opened_nodes
                        and next_node not in queque
                        and len(self.compute_path(next_node, pre, start)) <= d
                    ):
                        news.append(next_node)
                    elif old == -1:
                        pre.pop(next_node)
                    else:
                        pre[next_node] = old

                news.sort()
                queque = news + queque

        return [], opened_nodes

    def compute_path(self, last_node, pre, start):
        ins = last_node
        path = []
        while ins != start:
            path = [ins] + path
            ins = pre[ins]
        return [start] + path
