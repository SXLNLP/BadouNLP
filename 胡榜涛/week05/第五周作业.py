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


def main():
    model = load_word2vec_model(r"D:\AI课程学习\week5 词向量及文本向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    centers_vectors = kmeans.cluster_centers_

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    vector_centor_distance={}
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for vector, label in zip(vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)         #同标签的向量放在一起
    for label, vectors in vector_label_dict.items():
        average_vector = 0
        print(label)
        print(vectors)
        for vector in vectors:
            average_vector +=np.sqrt(np.sum((np.array(vector, dtype=float)  - np.array(centers_vectors[label])) ** 2))
        average_vector /= len(vectors)
        vector_centor_distance[label] = average_vector

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    label_order=sorted(vector_centor_distance.values())
    for i in range(min(3, len(label_order))):
        for label, distance in vector_centor_distance.items():
            if distance==label_order[i]:
                lables_index=label
        print(f"类内距离第{i+1}是第{lables_index}类，平均距离为{label_order[i]}")
        print('对应的分类为：')
        for i in range(min(10, len(sentence_label_dict.get(lables_index)))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict.get(lables_index)[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

