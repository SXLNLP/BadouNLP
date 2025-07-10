#!/usr/bin/env python3
#coding: utf-8

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def calculate_average_distance(vectors, labels, n_clusters):
    average_distances = []
    for i in range(n_clusters):
        cluster_points = vectors[labels == i]  # 获取同一聚类的所有点
        if cluster_points.shape[0] == 0:
            average_distances.append(0)  # 如果聚类没有点，距离为0
            continue
        cluster_center = cluster_points.mean(axis=0)  # 计算聚类中心
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)  # 计算距离
        average_distance = distances.mean()  # 计算平均距离
        average_distances.append(average_distance)
    return average_distances

def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起

    # 计算类内平均距离
    average_distances = calculate_average_distance(vectors, kmeans.labels_, n_clusters)

    # 创建一个列表来存储聚类标签和对应的类内平均距离
    cluster_distance_info = [(label, average_distances[label]) for label in range(n_clusters)]

    # 按照类内平均距离进行排序
    cluster_distance_info.sort(key=lambda x: x[1])

    # 输出聚类结果
    for label, avg_distance in cluster_distance_info:
        sentences = sentence_label_dict[label]
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("类内平均距离: ", avg_distance)  # 输出类内平均距离
        print("---------")

if __name__ == "__main__":
    main()
