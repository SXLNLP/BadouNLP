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
import matplotlib.pyplot as plt

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

# 计算每个聚类簇内部的平均距离
def calculate_and_plt_inner_distance(vectors, labels, centers):
    n_clusters = centers.shape[0]
    cluster_distances = []

    for i in range(n_clusters):
        # 找到第i个簇的所有向量
        cluster_points = vectors[labels == i]

        if len(cluster_points) == 0:
            cluster_distances.append(0)
            continue

        # 计算这些向量到该簇中心的欧式距离
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)

        # 平均距离
        avg_distance = np.mean(distances)
        cluster_distances.append(avg_distance)

        print(f"簇{i}的样本数：{len(cluster_points)}, 距离：{avg_distance:.4f}")
    
    cluster_distances.sort()
   
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(cluster_distances)),cluster_distances)
    plt.xlabel("Cluster")
    plt.ylabel("Average distance")
    plt.show()
    return


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
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
    
    calculate_and_plt_inner_distance(vectors, kmeans.labels_, kmeans.cluster_centers_)
if __name__ == "__main__":
    main()

