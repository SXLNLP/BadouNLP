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
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
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
    model = load_word2vec_model(r"N:\八斗\上一期\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence(r"N:\八斗\上一期\第五周 词向量\week5 词向量及文本向量\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_list = list(sentences)  
    sentence_label_dict = defaultdict(list)
    label_vector_dict = defaultdict(list)

    for sentence, vector, label in zip(sentence_list, vectors, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        label_vector_dict[label].append(vector)

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    # 计算每个类的类内平均距离
    cluster_avg_distances = {}
    for label, vecs in label_vector_dict.items():
        vecs = np.array(vecs)
        # center = kmeans.cluster_centers_[label]  
        # distances = np.linalg.norm(vecs - center, axis=1)   # 欧氏距离
        center = kmeans.cluster_centers_[label].reshape(1, -1)  
        distances = cosine_distances(vecs, center)  # 余弦距离
        avg_distance = np.mean(distances)
        cluster_avg_distances[label] = avg_distance

    # 取距离最小的前5个类，打印每个类内的10句话，不够10句则全部打印
    top5 = sorted(cluster_avg_distances.items(), key=lambda x: x[1])[:5]
    print("\n类内平均距离最小的5个类：")
    for label, dist in top5:
        print(f"\nCluster {label} - Avg Intra Distance: {dist:.4f}")
        sentences_in_cluster = sentence_label_dict[label]
        for i in range(min(10, len(sentences_in_cluster))):
            print(sentences_in_cluster[i].replace(" ", ""))  # 去掉分词空格
        print("---------")


if __name__ == "__main__":
    main()

