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
from sklearn.metrics import pairwise_distances
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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
        vectors.append(vector / len(words))
    return np.array(vectors)


def calculate_cluster_distances(vectors, labels, cluster_centers):
    # 计算每个样本到其聚类中心的距离
    distances = []
    for i in range(len(vectors)):
        # 计算欧式距离
        dist = np.linalg.norm(vectors[i] - cluster_centers[labels[i]])
        distances.append(dist)
    return distances


def main():
    model = load_word2vec_model(r"D:\ailearn\learnpool\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算每个聚类的平均距离
    cluster_distances = defaultdict(list)
    for i in range(len(vectors)):
        cluster_distances[kmeans.labels_[i]].append(
            np.linalg.norm(vectors[i] - kmeans.cluster_centers_[kmeans.labels_[i]])
        )

    # 计算每个聚类的平均距离
    avg_distances = {}
    for cluster_id, distances in cluster_distances.items():
        avg_distances[cluster_id] = np.mean(distances)

    # 按照平均距离排序聚类
    sorted_clusters = sorted(avg_distances.items(), key=lambda x: x[1])

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 按照平均距离顺序输出聚类结果
    for cluster_id, avg_distance in sorted_clusters:
        print(f"cluster {cluster_id} (平均距离: {avg_distance:.4f}):")
        # cluster_sentences = sentence_label_dict[cluster_id]
        # for i in range(min(10, len(cluster_sentences))):  # 随便打印几个，太多了看不过来
        #     print(cluster_sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
