import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import community.community_louvain as community_louvain

def styled_heading(text):
    st.markdown(
        f"""
        <div style="background-color:#ff7e5f;
                    border-radius:10px;
                    padding:10px;
                    text-align:center;
                    color:white;
                    font-weight:bold;
                    font-size:18px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def colored_line():
    st.markdown("<hr style='border:none; height:3px; background:#ff7e5f; margin:10px 0;'>", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    if "Amazon-Products.csv" in file_path:
        df["discount_price"] = df["discount_price"].astype(str).str.split("₹", expand=True).get(1)
        df["actual_price"] = df["actual_price"].astype(str).str.split("₹", expand=True).get(1)

        df["discount_price"] = df["discount_price"].str.replace(',', '').astype(float)
        df["actual_price"] = df["actual_price"].str.replace(',', '').astype(float)

        df["main_category"] = df["main_category"].astype(str).str.replace(' ', '')
        df["sub_category"] = df["sub_category"].astype(str).str.replace(' ', '')

        df["discount_price"] = df["discount_price"].fillna(df["discount_price"].median())
        df["actual_price"] = df["actual_price"].fillna(df["actual_price"].median())

        df["ratings"] = pd.to_numeric(df["ratings"], errors="coerce")
        df["ratings"] = df["ratings"].fillna(df["ratings"].mean())

        columns_to_extract_tags_from = ["name", "main_category", "sub_category"]
        df["Tags"] = df[columns_to_extract_tags_from].fillna("").agg(" ".join, axis=1)

    return df

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

def content_based_recommendations(df, item_name, top_n=10):
    if item_name not in df['name'].values:
        return pd.DataFrame()
    
    if 'Tags' not in df.columns:
        st.warning("Колонка 'Tags' отсутствует в датасете. Проверьте данные.")
        return pd.DataFrame()
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['Tags'].fillna(""))

    n_components = 100  
    svd = TruncatedSVD(n_components=n_components)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix_content)
    
    nn = NearestNeighbors(n_neighbors=top_n + 1, metric="cosine")
    nn.fit(reduced_tfidf_matrix)
    
    item_index = df[df['name'] == item_name].index[0]
    distances, indices = nn.kneighbors([reduced_tfidf_matrix[item_index]])

    recommended_item_indices = indices[0][1:] 
    
    return df.iloc[recommended_item_indices][['name', 'main_category', 'sub_category', 'ratings', 'actual_price']]

def analyze_data(df):
    styled_heading("Общая информация о данных")
    st.write(f" **Всего записей в датасете:** {df.shape[0]}")
    st.write(f" **Число уникальных категорий:** {df['main_category'].nunique()}")
    st.write(f" **Число уникальных подкатегорий:** {df['sub_category'].nunique()}")

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        styled_heading(" Пропущенные значения")
        st.write(missing_values[missing_values > 0])
    else:
        st.success(" В датасете нет пропущенных значений!")
    
    styled_heading("Описание данных")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True) 
    st.dataframe(df.describe())
    
    duplicate_count = df.duplicated().sum()
    styled_heading("Дубликаты")
    st.write(f"**Количество дубликатов:** {duplicate_count}")

def plot_category_distribution(df):
    styled_heading("Распределение основных категорий")
    main_category_counts = df['main_category'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=main_category_counts.index, y=main_category_counts.values, palette="viridis")
    plt.xticks(rotation=75)
    plt.xlabel("Категории")
    plt.ylabel("Количество")
    plt.title("Распределение товаров по категориям")
    st.pyplot(plt)

def plot_ratings_distribution(df):
    styled_heading("⭐ Распределение рейтингов")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ratings'], bins=20, kde=True, color='#ff6b6b', alpha=0.7)
    plt.xlabel("Рейтинг")
    plt.ylabel("Частота")
    plt.title("Распределение оценок пользователей")
    st.pyplot(plt)

def plot_top_ratings(df):
    styled_heading("Топ-20 товаров по количеству оценок")
    rating_counts = df['no_of_ratings'].value_counts().nlargest(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="magma")
    plt.xlabel("Количество отзывов")
    plt.ylabel("Частота")
    plt.title("Топ-20 товаров по количеству отзывов")
    st.pyplot(plt)

def plot_top_subcategories(df):
    styled_heading("Топ-20 подкатегорий товаров")
    df['sub_category'] = df['sub_category'].str.replace('&', 'and').str.replace(',', '').str.replace("'", "").str.replace('-', '').str.title()
    subcat_counts = df['sub_category'].value_counts()
    top_20_subcats = subcat_counts.nlargest(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(y=top_20_subcats.index, x=top_20_subcats.values, palette="coolwarm")
    plt.xlabel("Количество")
    plt.ylabel("Подкатегории")
    plt.title("Топ-20 подкатегорий (горизонтальный график)")
    st.pyplot(plt)

def visualize_subgraph(graph, node, recommended_nodes):
    subgraph = graph.subgraph([node] + list(graph.neighbors(node)) + recommended_nodes)
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, node_size=50, node_color='lightblue', edge_color='gray', with_labels=True)
    plt.title(f'Подграф рекомендаций для узла {node}')
    st.pyplot(plt)

df = load_data("Amazon-Products.csv")
graph = load_graph("com-amazon.ungraph.txt")

st.sidebar.title('Меню')
page = st.sidebar.radio('Выберите страницу', ['Главная', 'Аналитика данных', 'Рекомендации на основе графа', 'Контентные рекомендации'])

if page == 'Главная':
    st.title('Добро пожаловать в рекомендательную систему Amazon')
    st.write("Выберите страницу в меню слева.")
    st.write("Выполнили Хисаметдинова Динара и Борисова Элина, K3341.")

elif page == 'Аналитика данных':
    st.title('📊 Анализ данных')
    analyze_data(df)
    
    colored_line()
    plot_category_distribution(df)
    plot_ratings_distribution(df)
    plot_top_ratings(df)
    plot_top_subcategories(df)

elif page == 'Рекомендации на основе графа':
    st.title('Рекомендательная система товаров Amazon')
    node = st.number_input('Введите номер узла', min_value=0, max_value=len(graph.nodes()) - 1, step=1)
    num_recommendations = st.slider('Количество рекомендаций', min_value=1, max_value=10, value=5)
    if st.button('Сгенерировать рекомендации'):
        if node in graph:
            recommended_nodes = recommend_for_node(graph, node, num_recommendations)
            st.write(f'Рекомендации для узла {node}: {recommended_nodes}')
            visualize_subgraph(graph, node, recommended_nodes)
        else:
            st.warning('Выбранный узел отсутствует в графе.')

elif page == 'Контентные рекомендации':
    st.title('📖 Контентные рекомендации')
    item_name = st.text_input('Введите название товара')
    top_n = st.slider('Количество рекомендаций', min_value=1, max_value=10, value=5)
    if st.button('Рекомендовать'):
        recommendations = content_based_recommendations(df, item_name, top_n)
        if not recommendations.empty:
            st.write(recommendations)
        else:
            st.warning('Товар не найден в базе данных.')
