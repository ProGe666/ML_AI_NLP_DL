import networkx as nx
import matplotlib.pyplot as plt


def search(start, graph, concat_func):
    path = [start]
    seen = set()

    while path:
        node = path.pop(0)

        if node in seen: continue

        successors = graph[node]
        print('I am standing on: {}, looking forward{}'.format(node, successors))

        path = concat_func(path, successors)

        seen.add(node)


def bfs_strategy():
    def _wrap(path, successors):
        return path + successors

    return _wrap


def dfs_strategy():
    def _wrap(path, successors):
        return successors + path

    return _wrap


def draw_graph(graph):
    g = nx.Graph(graph)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, node_size=400, font_size=40)
    plt.show()


def main():
    graph = {
        0: [1, 5],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3],
        5: [0, 6],
        6: [5, 7],
        7: [6]
    }

    draw_graph(graph)

    print('Breadth First Search Processing: ')
    search(0, graph, concat_func=bfs_strategy())
    print('')
    print('Depth First Search Processing: ')
    search(0, graph, concat_func=dfs_strategy())


if __name__ == '__main__':
    main()
