#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import operator
import re
import json
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


def distance(point1, point2):
    if len(point1) != len(point2):
        print(point1, point2)
        print(len(point1), len(point2))
        print('不能计算距离！')
        return
    sum_distance = 0.0
    for num1, num2 in zip(point1, point2):
        sum_distance += pow(num1 - num2, 2)
    return pow(sum_distance, 0.5)


def main():
    model = load_word2vec_model(r"model.w2v")  #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)  #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)  #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)  #同标签的放到一起
    # TODO
    vectors_label_dict = defaultdict(list)
    distance_ave_label_dict = defaultdict(float)
    distance_vectors_center_dict = defaultdict(dict)
    cluster_centers = kmeans.cluster_centers_  # 获取中心点向量
    for vectors, label in zip(vectors, kmeans.labels_):  #取出向量和标签
        vectors_label_dict[label].append(vectors)  #同标签的放到一起
    for label, vectors in vectors_label_dict.items():
        sum_distance = 0.0
        for index in range(len(vectors)):
            vector = vectors[index]
            p2p_distance: float = distance(vector, cluster_centers[label])
            distance_vectors_center_dict[label][index] = p2p_distance
            sum_distance += p2p_distance
        distance_ave_label_dict[label] = sum_distance / len(vectors)
    sort_distance_ave_label_dict = dict(
        sorted(distance_ave_label_dict.items(), key=operator.itemgetter(1)))  # 将每个标签按与中心的平均距离排序
    sort_distance_vectors_center_dict = {
        key: dict(sorted(inner_dict.items(), key=lambda item: item[1]))
        for key, inner_dict in distance_vectors_center_dict.items()
    }  # 将每个标签内部的向量数据按照与中心距离排序
    print(sort_distance_ave_label_dict)

    for label, distance_ave in sort_distance_ave_label_dict.items():
        sentences = sentence_label_dict[label]
        min_distance_vectors_center_dict: dict = sort_distance_vectors_center_dict[label]  # 取出排好序的标签数据
        keyList: list = list(min_distance_vectors_center_dict.keys())
        print("cluster %s  average_distance %f :" % (label, distance_ave))
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""), end="       ")
            print("距离中心点:%f" % min_distance_vectors_center_dict[keyList[i]])
        print("---------")


if __name__ == "__main__":
    main()
