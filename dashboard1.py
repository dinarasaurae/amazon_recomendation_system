import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import community as community_louvain


@st.cache_data
def load_graph(file_path, max_nodes=10000, min_degree=5):
    graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
    nodes = [node for node, degree in graph.degree() if degree >= min_degree]
    return graph.subgraph(nodes[:max_nodes])


def recommend_for_node(graph, node, num_recommendations=5, max_neighbors=10):
    neighbors = sorted(set(graph.neighbors(node)), key=lambda n: graph.degree(n), reverse=True)[:max_neighbors]
    common_neighbors = defaultdict(int)

    for neighbor in neighbors:
        for n_neighbor in graph.neighbors(neighbor):
            if n_neighbor != node and n_neighbor not in neighbors:
                common_neighbors[n_neighbor] += 1

    recommendations = sorted(common_neighbors.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in recommendations[:num_recommendations]]


def community_detection(graph):
    partition = community_louvain.best_partition(graph)
    return partition


def pagerank(graph):
    pr = nx.pagerank(graph)
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)


def visualize_subgraph(graph, node, recommended_nodes):
    subgraph = graph.subgraph([node] + list(graph.neighbors(node)) + recommended_nodes)
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, node_size=50, node_color='lightblue', edge_color='gray', with_labels=True)
    plt.title(f"Подграф рекомендаций для узла {node}")
    st.pyplot(plt)


def main():
    file_path = 'com-amazon.ungraph.txt'
    max_nodes = 10000
    min_degree = 5
    graph = load_graph(file_path, max_nodes, min_degree)

    st.title('Рекомендательная система товаров Amazon')
    st.write(
        f"Количество узлов: {graph.number_of_nodes()}, количество рёбер: {graph.number_of_edges()}")
    node = st.number_input('Введите номер узла', min_value=0, max_value=len(graph.nodes()) - 1, step=1)
    num_recommendations = st.slider('Количество рекомендаций', min_value=1, max_value=10, value=5)
    pr = pagerank(graph)
    partition = community_detection(graph)
    st.write(f"Сообщество для узла {node}: {partition.get(node)}")

    if st.button("Сгенерировать рекомендации"):
        if node in graph:
            recommended_nodes = recommend_for_node(graph, node, num_recommendations)
            st.write(f"Рекомендации для узла {node}: {recommended_nodes}")
            visualize_subgraph(graph, node, recommended_nodes)
        else:
            st.warning("Выбранный узел отсутствует в графе.")


if __name__ == "__main__":
    main()