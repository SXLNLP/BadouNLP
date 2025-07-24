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
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)  # 42
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    sentence_label_dict = defaultdict(list)
    
    # 存储每个簇的句子和向量
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        sentence_label_dict[label].append((sentence, vector))
    
    # 计算每个簇的平均距离并排序
    cluster_info = {}
    for label, items in sentence_label_dict.items():
        cluster_points = vectors[kmeans.labels_ == label]
        center = kmeans.cluster_centers_[label]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        mean_distance = np.mean(distances)
        cluster_info[label] = {
            'sentences': items,
            'mean_distance': mean_distance,
            'size': len(items)
        }
    
    # 按平均距离从大到小排序
    sorted_clusters = sorted(cluster_info.items(), 
                           key=lambda x: x[1]['mean_distance'], 
                           reverse=True)
    
    # 打印前10个簇的信息
    print("\n聚类结果(按平均距离排序):")
    for i, (label, info) in enumerate(sorted_clusters):
        if i >= 10:
            break
        print(f"\n簇{label} (平均距离: {info['mean_distance']:.4f}, 包含{info['size']}个句子):")
        for sentence, _ in info['sentences'][:5]:  # 只显示前5个句子
            print(f"  - {sentence}")
    
if __name__ == "__main__":
    main()
