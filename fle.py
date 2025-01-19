import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities


graph = nx.read_edgelist("com-amazon.ungraph.txt", nodetype=int)

subgraph_nodes = list(graph.nodes())[:15000]
subgraph = graph.subgraph(subgraph_nodes)
mutable_subgraph = nx.Graph(subgraph)
mutable_subgraph.remove_nodes_from(list(nx.isolates(mutable_subgraph)))

plt.figure(figsize=(10, 10))
nx.draw(mutable_subgraph, node_size=10, alpha=0.5)
plt.show()
print(f"Nodes in subgraph: {mutable_subgraph.number_of_nodes()}, Edges in subgraph: {mutable_subgraph.number_of_edges()}")

degree_centrality = nx.degree_centrality(mutable_subgraph)
betweenness_centrality = nx.betweenness_centrality(mutable_subgraph)

from networkx.algorithms.community import greedy_modularity_communities
clusters = list(greedy_modularity_communities(mutable_subgraph))
print(f"Number of clusters in subgraph: {len(clusters)}")

pagerank = nx.pagerank(mutable_subgraph)

node = list(mutable_subgraph.nodes())[0]
neighbors = list(mutable_subgraph.neighbors(node))[:100]
recommendations = sorted(neighbors, key=lambda x: pagerank[x], reverse=True)[:5]
print(f"Recommendations for {node}: {recommendations}")

# Гипотеза 1
degree_centrality = nx.degree_centrality(mutable_subgraph)
betweenness_centrality = nx.betweenness_centrality(mutable_subgraph)

# 5 узлов с наибольшей Degree Centrality
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 nodes by Degree Centrality:")
for node, centrality in top_degree:
    print(f"Node: {node}, Degree Centrality: {centrality:.4f}")

# 5 узлов с наибольшей Betweenness Centrality
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 nodes by Betweenness Centrality:")
for node, centrality in top_betweenness:
    print(f"Node: {node}, Betweenness Centrality: {centrality:.4f}")

# Гипотеза 2

# Кластеры с помощью Greedy Modularity
clusters = list(greedy_modularity_communities(mutable_subgraph))
print(f"\nNumber of clusters: {len(clusters)}")

for i, cluster in enumerate(clusters[:5]):
    print(f"Cluster {i + 1}: {len(cluster)} nodes")
color_map = []
node_cluster_map = {}
for cluster_id, cluster in enumerate(clusters):
    for node in cluster:
        node_cluster_map[node] = cluster_id

for node in mutable_subgraph.nodes():
    color_map.append(node_cluster_map[node])
plt.figure(figsize=(10, 10))
nx.draw(mutable_subgraph, node_size=10, node_color=color_map, cmap='viridis', alpha=0.7)
plt.title("Clusters in Subgraph")
plt.show()

# Гипотеза 3
# Link Prediction - Jaccard Similarity
from itertools import combinations
from networkx.algorithms.link_prediction import jaccard_coefficient

# Считаем коэффициент Жаккара для случайной выборки пар узлов
pairs = list(combinations(mutable_subgraph.nodes(), 2))[:100]
jaccard_scores = list(jaccard_coefficient(mutable_subgraph, pairs))

# Выводим первые 5 результатов
print("\nSample Jaccard Similarity Scores:")
for u, v, score in jaccard_scores[:5]:
    print(f"Nodes ({u}, {v}): Jaccard Score = {score:.4f}")

# Гипотеза 4
# PageRank
pagerank = nx.pagerank(mutable_subgraph)

# Топ-5 узлов с наибольшим PageRank
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 nodes by PageRank:")
for node, rank in top_pagerank:
    print(f"Node: {node}, PageRank: {rank:.4f}")
