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


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 存储每个簇的句子及其向量
    cluster_sentences = defaultdict(list)
    cluster_vectors = defaultdict(list)

    # 同时收集句子和对应的向量
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        cluster_sentences[label].append(sentence)
        cluster_vectors[label].append(vector)

    # 计算每个簇的中心点
    centers = kmeans.cluster_centers_

    # 存储每个簇的排序结果
    sorted_clusters = defaultdict(list)

    # 对每个簇内的样本按距离排序
    for label in cluster_sentences.keys():
        # 获取当前簇的中心点
        center = centers[label]
        # 计算当前簇内每个样本到中心的距离
        distances = []
        for vector in cluster_vectors[label]:
            # 使用欧氏距离
            dist = np.linalg.norm(vector - center)
            distances.append(dist)

        # 将距离和句子组合，并按距离排序
        sentence_distance_pairs = list(zip(cluster_sentences[label], distances))
        # 按距离从小到大排序（距离小表示更接近中心）
        sorted_sentences = sorted(sentence_distance_pairs, key=lambda x: x[1])
        sorted_clusters[label] = sorted_sentences

    # 输出排序后的聚类结果
    for label, sorted_sentences in sorted_clusters.items():
        print("cluster %s :" % label)
        # 打印前10个句子（按距离从近到远）
        for i in range(min(10, len(sorted_sentences))):
            sentence, distance = sorted_sentences[i]
            print(f"{sentence.replace(' ', '')} (距离: {distance:.4f})")
        print("---------")


if __name__ == "__main__":
    main()
