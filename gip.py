import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index
from sklearn.metrics import precision_score, recall_score

# Загрузите граф
graph = nx.read_edgelist("com-amazon.ungraph.txt", nodetype=int)

# Извлекаем подграф для обработки
subgraph_nodes = list(graph.nodes())[:500]  # Берём только 500 узлов для теста
subgraph = graph.subgraph(subgraph_nodes)
mutable_subgraph = nx.Graph(subgraph)
mutable_subgraph.remove_nodes_from(list(nx.isolates(mutable_subgraph)))

# 1. Link Prediction - Jaccard Similarity
pairs = list(combinations(mutable_subgraph.nodes(), 2))[:1000]  # Увеличиваем выборку до 1000 пар узлов
jaccard_scores = list(jaccard_coefficient(mutable_subgraph, pairs))

# 2. Link Prediction - Common Neighbors (Ручная реализация)
common_neighbors_scores = []
for u, v in pairs:
    common_neighbors_score = len(list(nx.common_neighbors(mutable_subgraph, u, v)))
    common_neighbors_scores.append((u, v, common_neighbors_score))

# 3. Link Prediction - Adamic-Adar
adamic_adar_scores = list(adamic_adar_index(mutable_subgraph, pairs))

# Считаем базовые метрики Precision и Recall
# Разделим данные на обучающую и тестовую выборки
train_edges = list(mutable_subgraph.edges())[:int(len(mutable_subgraph.edges()) * 0.8)]
test_edges = list(mutable_subgraph.edges())[int(len(mutable_subgraph.edges()) * 0.8):]

# Функция для оценки метрик
def calculate_metrics(predicted_edges, test_edges):
    y_true = [1 if edge in test_edges else 0 for edge in predicted_edges]
    y_pred = [1 for edge in predicted_edges]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall

# Пример для Jaccard Similarity
jaccard_predicted_edges = [(u, v) for u, v, score in jaccard_scores if score > 0.01]  # Порог для предсказаний
precision_jaccard, recall_jaccard = calculate_metrics(jaccard_predicted_edges, test_edges)

# Пример для Common Neighbors
common_neighbors_predicted_edges = [(u, v) for u, v, score in common_neighbors_scores if score > 0]
precision_common_neighbors, recall_common_neighbors = calculate_metrics(common_neighbors_predicted_edges, test_edges)

# Пример для Adamic-Adar
adamic_adar_predicted_edges = [(u, v) for u, v, score in adamic_adar_scores if score > 0]
precision_adamic_adar, recall_adamic_adar = calculate_metrics(adamic_adar_predicted_edges, test_edges)

# Выводим результаты
print("Jaccard Similarity:")
print(f"Precision: {precision_jaccard:.4f}")
print(f"Recall: {recall_jaccard:.4f}")

print("\nCommon Neighbors:")
print(f"Precision: {precision_common_neighbors:.4f}")
print(f"Recall: {recall_common_neighbors:.4f}")

print("\nAdamic-Adar:")
print(f"Precision: {precision_adamic_adar:.4f}")
print(f"Recall: {recall_adamic_adar:.4f}")

# Визуализируем граф с предсказанными связями (если необходимо)
color_map = ['blue' if node in mutable_subgraph.nodes() else 'red' for node in graph.nodes()]
plt.figure(figsize=(10, 10))
nx.draw(mutable_subgraph, node_size=10, node_color=color_map, alpha=0.7)
plt.title("Graph with Predicted Links")
plt.show()