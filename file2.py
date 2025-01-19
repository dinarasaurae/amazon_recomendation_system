import networkx as nx
import random
file_path = 'com-amazon.ungraph.txt'
graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)

def recommend_for_node(graph, node, num_recommendations=5):
    """
    Рекомендует узлы, основываясь на соседях заданного узла.

    :param graph: Исходный граф.
    :param node: Узел, для которого требуется рекомендация.
    :param num_recommendations: Количество рекомендаций.
    :return: Список рекомендованных узлов.
    """
    neighbors = set(graph.neighbors(node))
    common_neighbors = {}

    for neighbor in neighbors:
        for n_neighbor in graph.neighbors(neighbor):
            if n_neighbor != node and n_neighbor not in neighbors:
                if n_neighbor in common_neighbors:
                    common_neighbors[n_neighbor] += 1
                else:
                    common_neighbors[n_neighbor] = 1
    recommendations = sorted(common_neighbors.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in recommendations[:num_recommendations]]
random_node = random.choice(list(graph.nodes()))
recommended_nodes = recommend_for_node(graph, random_node, num_recommendations=5)

print(f"Рекомендации для узла {random_node}: {recommended_nodes}")