# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/9 15:16
@Auth ： fengbangwei
@File ：homework_kmeans.py

"""

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
    实现基于kmeans结果类内距离的排序。
"""


# 加载文本
def load_text(file_path):
    sentences = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    sentences = sorted(sentences)
    return sentences


# 加载模型
def load_model(file_path):
    model = Word2Vec.load(file_path)
    return model


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def get_sorted_labels(sentences, kmeans, vectors):
    # 文本 对应 分类标签
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    # 每个向量对应的类别
    vectors_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        vectors_label_dict[label].append(vector)

    # 每个类别中心向量
    dist_center_avg_dict = {}
    for label, center_vector in enumerate(kmeans.cluster_centers_):
        vectors = vectors_label_dict[label]
        # 计算平均距离
        dist_center_avg = np.mean([np.linalg.norm(center_vector - vector) for vector in vectors])
        dist_center_avg_dict[label] = dist_center_avg

    # 按平均距离排序 升序
    dist_center_avg_dict_sort = sorted(dist_center_avg_dict.items(), key=lambda x: x[1], reverse=False)
    # print(dist_center_avg_dict_sort)
    return vectors_label_dict, sentence_label_dict, dist_center_avg_dict_sort


def plot_clusters(vectors, labels, centers, title="All Cluster"):
    # 使用 PCA 降维到二维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # 绘制每个样本点
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idxs = np.where(labels == label)
        plt.scatter(vectors_2d[idxs, 0], vectors_2d[idxs, 1], label=f'Cluster {label}', s=50)

    # 绘制聚类中心
    centers_2d = pca.transform(centers)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, c='black', marker='X', label='Centers')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    sentences = load_text("titles.txt")
    # print(sentences)
    model = load_model("3_model.w2v")
    vectors = sentences_to_vectors(sentences, model)

    # 设置聚类数量
    n_clusters = int(np.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters, init='k-means++', random_state=42)
    kmeans.fit(vectors)

    print(vectors.shape)
    print(kmeans.labels_.shape)
    print(kmeans.cluster_centers_.shape)
    # 可视化聚类结果
    # plot_clusters(vectors, kmeans.labels_, kmeans.cluster_centers_)

    # dist_avg_dict_sort (类别，平均距离)
    vectors_label_dict, sentence_label_dict, dist_center_avg_dict_sort = get_sorted_labels(sentences, kmeans, vectors)

    # 在聚类完成后，先计算每个类别的内部样本与中心的平均距离，再按距离大小排序，最后剔除掉那些类内样本分布较散、质量较差的类别。
    # 类内平均距离越大，说明这个类内的样本越分散，聚类效果不好。
    # 可能表示该类别包含噪声、异常值或不适合聚在一起的数据。
    # 舍弃这些类别有助于后续分析中保留“高质量”的聚类结果。
    # 取前20个 平均距离小的类别
    top20_labels = dist_center_avg_dict_sort[:20]
    print(top20_labels)
    for label, sentences in sentence_label_dict.items():
        # 在top20中 则输出
        if label in [x[0] for x in top20_labels]:
            print("cluster %s :" % label)
            for i in range(min(10, len(sentences))):
                print(sentences[i].replace(" ", ""))
            print("---------")

    new_vectors = []
    new_labels = []
    new_centers = []
    for label, center_vector in enumerate(kmeans.cluster_centers_):
        if label in [x[0] for x in top20_labels]:
            # vector list
            vectors = list(vectors_label_dict[label])
            for vector in vectors:
                new_vectors.append(vector)
                new_labels.append(label)
            new_centers.append(center_vector)

    print(np.array(new_vectors).shape)
    print(np.array(new_labels).shape)
    print(np.array(new_centers).shape)
    # 可视化聚类结果
    plot_clusters(new_vectors, new_labels, new_centers, title="Top20 Cluster")


if __name__ == '__main__':
    main()
