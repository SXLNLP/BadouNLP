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
    model = load_word2vec_model(r"D:\八斗课程视频\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 获取聚类质心的坐标
    centroids = kmeans.cluster_centers_
    print(f'质心坐标维度：{centroids.shape}')

    # 创建字典存储每个标签的句子、向量和距离
    cluster_data = defaultdict(list)

    # 计算每个句子的距离并存储
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        centroid = centroids[label]
        distance = np.linalg.norm(vector - centroid)
        cluster_data[label].append((sentence, distance))

    # 对每个聚类内的句子按距离升序排序（距离小的在前）
    for label in cluster_data:
        cluster_data[label].sort(key=lambda x: x[1])

    # 输出距离排序的结果
    for label, data_list in cluster_data.items():
        print(f'\n cluster {label} 按距离升序排序：')
        # 近输出十个查看
        for i, (sentence, distance) in enumerate(data_list):
            if i >= 10:
                break
            # 输出句子（去除分词空格）和距离（保留4位小数）
            print(f'{i + 1}.{sentence.replace(' ', '')} (距离：{distance:.4f})')
        print('------------')


    # sentence_label_dict = defaultdict(list)
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()
