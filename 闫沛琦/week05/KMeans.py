#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model

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
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    sentences = []
    with open("corpus.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line))
    train_word2vec_model(sentences, 100)

    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 存储每个簇内点的距离排序结果
    cluster_distances = {}

    # 遍历每一个簇
    for cluster_id in range(kmeans.n_clusters):
        # 获取当前簇的所有样本及其索引
        cluster_mask = (labels == cluster_id)
        cluster_samples = vectors[cluster_mask]

        # 计算每个样本到中心点的欧氏距离
        distances = np.linalg.norm(cluster_samples - centroids[cluster_id], axis=1)

        # 按距离升序排列并获取排序索引
        sorted_indices = np.argsort(distances)

        # # 保存排序结果（包括距离和样本）
        cluster_distances[cluster_id] = {
            'samples': cluster_samples[sorted_indices],
            'distances': distances[sorted_indices]
        }

    # 打印每个簇的距离排序结果
    for cluster_id in cluster_distances:
        print(f"\nCluster {cluster_id} 的排序距离示例：")
        print("前5个最近的距离:", cluster_distances[cluster_id]['distances'][:5])
        print("对应样本向量:", cluster_distances[cluster_id]['samples'][:5])
        # print("前5个最远的距离:", cluster_distances[cluster_id]['distances'][-5:])


if __name__ == "__main__":
    main()

