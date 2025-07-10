#!/usr/bin/env python3  
#coding: utf-8

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
    # 添加 mmap=None 参数避免版本兼容问题
    model = Word2Vec.load(path, mmap=None)
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
        word_count = 0
        for word in words:
            try:
                vector += model.wv[word]
                word_count += 1
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        if word_count > 0:
            vector /= word_count
        vectors.append(vector)
    return np.array(vectors)

def main():
    # 修改路径
    model_path = r"F:\BaiduNetdiskDownload\八斗精品班\第五周\八斗精品班\week5 词向量及文本向量\week5 词向量及文本向量\model.w2v"
    model = load_word2vec_model(model_path)  # 加载词向量模型
    
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算到聚类中心的距离矩阵
    distances = kmeans.transform(vectors)

    # 构建标签到句子及其距离的映射关系
    sentence_label_dict = defaultdict(list)
    for i, (sentence, label) in enumerate(zip(sentences, kmeans.labels_)):
        distance = distances[i][label]  # 获取当前样本到对应聚类中心的距离
        sentence_label_dict[label].append((sentence, distance))

    # 按类内距离排序并输出
    for label, data in sentence_label_dict.items():
        print(f"cluster {label}:")
        sorted_data = sorted(data, key=lambda x: x[1])
        for i in range(min(10, len(sorted_data))): 
            print(sorted_data[i][0])
        print("---------")

if __name__ == "__main__":
    main()
