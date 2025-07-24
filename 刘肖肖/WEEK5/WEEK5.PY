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
from scipy.spatial.distance import pdist, squareform

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

# 计算类内平均距离
def calculate_intra_cluster_distance(cluster_vectors):
    if len(cluster_vectors) < 2:
        return 0
    distances = pdist(cluster_vectors)
    return np.mean(distances)

def main():
    model = load_word2vec_model(r"H:\AI\AI 八斗\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  #取出句子、向量和标签
        sentence_label_dict[label].append(sentence)         #同标签的句子放到一起
        vector_label_dict[label].append(vector)             #同标签的向量放到一起

    # 计算每个聚类的类内平均距离
    intra_cluster_distances = {}
    for label, cluster_vectors in vector_label_dict.items():
        distance = calculate_intra_cluster_distance(np.array(cluster_vectors))
        intra_cluster_distances[label] = distance

    # 按类内平均距离对聚类进行排序
    sorted_clusters = sorted(intra_cluster_distances.items(), key=lambda item: item[1], reverse=True)

    for label, distance in sorted_clusters:
        print(f"cluster {label} : 类内平均距离 = {distance}")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
