#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words) if len(words) > 0 else vector)
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init=10)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算每个样本到其所属类中心的距离
    distances = euclidean_distances(vectors, kmeans.cluster_centers_[kmeans.labels_])
    sample_distances = distances[np.arange(len(vectors)), kmeans.labels_]

    # 组织数据结构：记录每个类别的所有样本索引、距离及类内距离
    class_info = {}
    for label in range(n_clusters):
        # 获取当前类别的样本索引
        indices = np.where(kmeans.labels_ == label)[0]
        class_vectors = vectors[indices]

        # 计算类内平均距离
        if len(class_vectors) > 0:
            centroid = kmeans.cluster_centers_[label]
            intra_distances = np.linalg.norm(class_vectors - centroid, axis=1)
            avg_distance = np.mean(intra_distances)
        else:
            avg_distance = 0.0

        class_info[label] = {
            'indices': indices,
            'distances': sample_distances[indices],
            'avg_distance': avg_distance
        }

    # 按类内平均距离升序排序（类内距离越小表示类越紧密）
    sorted_classes = sorted(class_info.items(), key=lambda x: x[1]['avg_distance'])

    # 输出结果
    for class_label, info in sorted_classes:
        print(f"cluster {class_label}: (类内平均距离: {info['avg_distance']:.4f})")

        # 获取排序后的样本(按距离类中心的距离升序排列)
        sorted_indices = info['indices'][np.argsort(info['distances'])]

        for i in range(min(10, len(sorted_indices))):  # 每个类最多显示10条
            idx = sorted_indices[i]
            print(
                f"  (距离: {info['distances'][np.where(info['indices'] == idx)[0][0]]:.4f}) {sentences[idx].replace(' ', '')}")
        print("---------")


if __name__ == "__main__":
    main()
