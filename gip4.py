import networkx as nx

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

degree_centrality = nx.degree_centrality(mutable_subgraph)
betweenness_centrality = nx.betweenness_centrality(mutable_subgraph)
clusters = list(greedy_modularity_communities(mutable_subgraph))

def cluster_density(cluster, graph):
    subgraph = graph.subgraph(cluster)
    num_edges = subgraph.number_of_edges()
    num_nodes = len(cluster)
    return num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

intra_cluster_densities = [cluster_density(cluster, mutable_subgraph) for cluster in clusters]
inter_cluster_edges = 0
for i, cluster1 in enumerate(clusters):
    for cluster2 in clusters[i + 1:]:
        inter_cluster_edges += len(set(cluster1) & set(cluster2))

avg_intra_cluster_density = sum(intra_cluster_densities) / len(intra_cluster_densities)
print(f"Average intra-cluster density: {avg_intra_cluster_density:.4f}")
print(f"Number of inter-cluster edges: {inter_cluster_edges}")