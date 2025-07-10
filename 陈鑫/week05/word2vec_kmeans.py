#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"D:\八斗NLP课件\week05 词向量及文本向量\model.w2v")    # 加载词向量模型
    sentences = load_sentence(r"D:\八斗NLP课件\week05 词向量及文本向量\titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # vectors : 1796个句子向量 (1796, 128)
    # sentences : 1796个分词后的句子(1796, )
    # kmeans.labels_ : 1796个句子的标签(1796, )
    # kmeans.cluster_centers_ : 聚类的中心(42, 128)

    # 构造{标签-句子向量列表}的字典
    label_vector_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        label_vector_dict[label].append(vector)

    # 计算类内平均距离 - 欧式距离
    label_avg_distance = dict()
    for label, vector in label_vector_dict.items():     # vector : n个label标签的句子向量
        cluster_center = kmeans.cluster_centers_[label]

        # 方式1. 直接使用numpy的linalg.norm函数
        # avg_distance = np.mean(np.linalg.norm(vector - cluster_center, axis=1))   # 直接计算每行的L2范数，然后求平均
        # 方式2. 手动计算
        avg_distance = np.mean(np.sqrt(np.sum((vector - cluster_center) ** 2, axis=1)))
        label_avg_distance[label] = avg_distance

    # 按类内平均距离进行排序
    sorted_label_distance = sorted(label_avg_distance.items(), key=lambda x: x[1])
    for x in sorted_label_distance:
        print(f"cluster:{x[0]},\t avg_distance:{x[1]:.4f}")


if __name__ == "__main__":
    main()

