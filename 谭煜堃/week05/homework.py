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

class SentenceCluster:
    def __init__(self, label):
        self.label = label
        self.sentences = []
        self.vectors = []

    def __str__(self):
        return f"-----\nCluster {self.label}: {self.sentences[:3]} \n mean_distance_to_centroid: {self.mean_distance_to_centroid}\n----- "

    def add_sentence(self, sentence, vector):
        self.sentences.append(sentence)
        self.vectors.append(vector)
    def get_centroid(self):
        return np.mean(self.vectors, axis=0)
    def get_mean_distance_to_centroid(self):
        centroid = self.get_centroid()
        return np.mean(np.linalg.norm(self.vectors - centroid, axis=1))
    def inject_mean_distance_to_centroid(self):
        self.mean_distance_to_centroid = self.get_mean_distance_to_centroid()
    # def get_max_distance_to_centroid(self):
    #     centroid = self.get_centroid()
    #     return np.max(np.linalg.norm(self.vectors - centroid, axis=1))


def main():
    """作业：实现聚类并按类内平均距离排序"""
    model = load_word2vec_model(r".\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    sentence_clusters = [SentenceCluster(i) for i in range(n_clusters)]

    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  #取出句子和标签
        sentence_clusters[label].add_sentence(sentence, vector)         #同标签的放到一起

    # 外部注入类内平均距离
    for cluster in sentence_clusters:
        cluster.inject_mean_distance_to_centroid() #外部注入类内平均距离
    
    # 按类内平均距离排序
    sentence_clusters.sort(key=lambda x: x.mean_distance_to_centroid)
    for cluster in sentence_clusters:
        print(cluster)
    
    # 打印前10个类
    

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

