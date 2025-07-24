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
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))

    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = [] #为保证顺序，用列表保存
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
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++",n_init=20, random_state=42)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    print("聚类结果：", kmeans.labels_)
    #根据聚类结果进行标题向量分组
    vector_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):  #取出句子向量和标签
        vector_label_dict[label].append(vector)         #同标签的放到一起

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    
    #设定长度为42的列表记录每个类别对应的平均距离
    avg_distances = [0] * n_clusters
    #循环计算每个类别的句子与类中心点的平均距离
    for label, vectors in vector_label_dict.items():
        center = kmeans.cluster_centers_[label]
        distances = np.sum((vectors - center) ** 2, axis=1)
        avg_distance = distances.mean()
        avg_distances[label] = avg_distance
        print("聚类 %d 的平均距离：%f" % (label, avg_distance))
    
    #排序取出距离最小的前10个类别
    sorted_labels = sorted(range(n_clusters), key=lambda x: avg_distances[x])
    print("距离最小的前10个类别：", sorted_labels[:10])
    #打印距离最小的前10个类别对应的前10个句子
    for label in sorted_labels[:10]:
        print("聚类 %d 的前10个句子：" % label)
        for sentence in sentence_label_dict[label][:10]:
            print(sentence.replace(" ", ""))
        print("-----------------------------------------------")
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

