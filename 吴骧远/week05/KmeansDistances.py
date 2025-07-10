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
        words = sentence.split()  # sentence是分好词的，空格分开KmeansDistances
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


# 计算类内平均距离
def calculate_intra_cluster_distances(vectors, labels, cluster_centers):
    cluster_distances = {}

    # 对每个聚类计算类内平均距离
    for cluster_id in set(labels):
        # 获取属于当前聚类的所有点
        cluster_mask = labels == cluster_id
        cluster_points = vectors[cluster_mask]
        cluster_center = cluster_centers[cluster_id]

        # 计算每个点到聚类中心的欧式距离
        distances = []
        for point in cluster_points:
            distance = np.linalg.norm(point - cluster_center)
            distances.append(distance)

        # 计算该聚类的平均距离
        if len(distances) > 0:
            avg_distance = np.mean(distances)
            cluster_distances[cluster_id] = {
                'avg_distance': avg_distance,
                'num_points': len(cluster_points)
            }

    return cluster_distances

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    cluster_distances = calculate_intra_cluster_distances(
        vectors, kmeans.labels_, kmeans.cluster_centers_
    )

    # 按类内平均距离排序
    sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1]['avg_distance'])

    # 输出每个聚类的统计信息
    print(f"{'clusterID':<10} {'样本数量':<12} {'平均距离':<12}")
    for cluster_id, info in sorted_clusters:
        avg_dist = info['avg_distance']
        num_points = info['num_points']

        print(f"{cluster_id:<12} {num_points:<12} {avg_dist:<12.4f}")

if __name__ == "__main__":
    main()
