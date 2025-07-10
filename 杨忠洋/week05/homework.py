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


def calculate(vector, param, type="euclidean") -> float:
    if type == "euclidean":
        # 欧式距离
        return np.sqrt(np.sum(np.square(param - vector)))
    else:
        # 余弦距离
        return 1 - np.dot(vector, param) / (np.linalg.norm(vector) * np.linalg.norm(param))


def main():
    model = load_word2vec_model("model.w2v")  #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)  #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)  #进行聚类计算

    sentence_label_dict = defaultdict(list)
    label_mean_distance_dict = defaultdict(float)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  #取出句子和标签
        sentence_label_dict[label].append(sentence)  #同标签的放到一起
        sentence_label_distance = calculate(vector, kmeans.cluster_centers_[label])
        label_mean_distance_dict[label] += sentence_label_distance
    # 计算类内平均距离
    for label in label_mean_distance_dict:
        label_mean_distance_dict[label] /= len(sentence_label_dict[label])
    # 排序
    label_mean_distance_dict = sorted(label_mean_distance_dict.items(), key=lambda x: x[1], reverse=True)
    for label, mean_distance in label_mean_distance_dict:
        print("cluster %s mean distance : %s" % (label, mean_distance))
    # 舍弃掉类内平均距离最大的n_clusters/10 个类
    for i in range(int(n_clusters / 10)):
        label = label_mean_distance_dict[i][0]
        sentence_label_dict.pop(label)

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
