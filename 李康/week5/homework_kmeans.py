#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial import distance
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

def DoKeyMeans(n_clusters, vectors):

    distances = defaultdict(list)
    sum_distance = 0
    # print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    center = kmeans.cluster_centers_  # 获取聚类中心坐标 大小[label_num, vector_size]
    for vector, label in zip(vectors, kmeans.labels_):  # 取出句子和标签
        cur_center = center[label]
        cur_distance = distance.euclidean(cur_center, vector)  # 使用欧式距离作为评价指标
        distances[label].append(cur_distance)
        sum_distance += cur_distance

    # distances = dict(sorted([(k, v) for k, v in distances.items()], key=lambda k: k[0]))
    distances = dict(sorted([(k, v) for k, v in distances.items()], key=lambda k: sum(k[1])))
    for label, dis in distances.items():
        print("label:", label, " dis_sum:", sum(dis))
    print("total distance:", sum_distance)
    return sum_distance

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    vectors = sentences_to_vectors(sentences, model) #将所有标题向量化
    DoKeyMeans(n_clusters, vectors)
    # sum_distances = []
    # for i_clusters in range(1, 100):
    #     sum_distances.append(DoKeyMeans(i_clusters, vectors))
    #
    # fig, ax = plt.subplots()
    # ax.plot(range(1, 100), sum_distances, linestyle='-', marker='o', markerfacecolor='r',
    #         markersize=10)
    # ax.set_title("cluster dis relationship", fontsize=20)
    # ax.set_xlabel("cluster", fontsize=20)
    # ax.set_ylabel("cluster dis sum", fontsize=20)
    # plt.show()

if __name__ == "__main__":
    main()

