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
        a = 0
        for d in range(len(graph)):
            queque = [start]
            opened_nodes = []
            pre = {}
            distance = {start: 0}

            while len(queque) != 0 and min(distance[x] for x in queque) <= d:
                node = queque[0]
                opened_nodes.append(node)
                queque.pop(0)

                if node == goal:
                    return self.compute_path(node, pre, start), opened_nodes

                news = []
                for next_node in graph[node]:
                    try:
                        if distance[next_node] >= (distance[node] + 1):
                            distance[next_node] = distance[node] + 1
                    except KeyError:
                        distance[next_node] = distance[node] + 1
                    if next_node not in opened_nodes and next_node not in queque and distance[next_node] < d:
                        news.append(
                            next_node
                        )  # if the node hasn't been opened yet and is distance is <= d so add in queque
                        pre[next_node] = node

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
