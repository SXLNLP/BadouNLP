#!/usr/bin/env python3
# coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split() #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 对每个簇内的距离进行排序
def calculate_and_sort_distances(vectors, labels, centers):
    # 存储每个簇的(距离, 句子索引)列表
    cluster_distances = defaultdict(list)

    for i, (vector, label) in enumerate(zip(vectors, labels)):
        distance = np.linalg.norm(vector - centers[label])
        cluster_distances[label].append((distance, i))

    # 对每个簇内的距离进行排序
    for label in cluster_distances:
        cluster_distances[label].sort(key=lambda x: x[0])

    return cluster_distances


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = list(load_sentence("titles.txt")) #加载所有标题
    vectors = sentences_to_vectors(sentences, model) #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences))) #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors) #进行聚类计算

    # 计算并排序每个簇内点到质心的距离
    cluster_distances = calculate_and_sort_distances(vectors, kmeans.labels_, kmeans.cluster_centers_)

    # 计算每个簇的平均距离用于排序
    cluster_avg_distances = {
        label: sum(dist for dist, _ in distances) / len(distances)
        for label, distances in cluster_distances.items()
    }

    # 按平均距离从大到小排序簇
    sorted_clusters = sorted(cluster_avg_distances.items(), key=lambda x: x[1], reverse=True)

    # 输出每个簇的信息
    for label, avg_distance in sorted_clusters:
        distances = cluster_distances[label]
        print(f"\ncluster {label} (平均距离: {avg_distance:.4f}, 样本数: {len(distances)})")
        print("=" * 50)

        print("\n所有样本距离质心长度排序:")
        for i, (distance, idx) in enumerate(distances, 1):
            print(f"{i}. [距离: {distance:.4f}] {sentences[idx].replace(' ', '')}")

        print("=" * 50)


if __name__ == "__main__":
    main()
