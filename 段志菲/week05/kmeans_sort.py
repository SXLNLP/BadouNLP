#!/usr/bin/env python3  
#coding: utf-8

# 基于训练好的词向量模型进行聚类，并按类内距离排序输出
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from collections import defaultdict

def load_word2vec_model(path):
    """加载训练好的Word2Vec模型"""
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    """加载文本数据并进行分词处理"""
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

def sentences_to_vectors(sentences, model):
    """将句子列表转换为向量矩阵"""
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

def calculate_intra_cluster_distances(vectors, labels, cluster_centers):
    """计算每个簇的类内平均距离"""
    intra_distances = []
    cluster_info = []
    
    for cluster_id in range(len(cluster_centers)):
        # 获取当前簇的所有向量
        cluster_vectors = vectors[labels == cluster_id]
        
        if len(cluster_vectors) == 0:
            intra_distances.append(0)
            cluster_info.append({
                "cluster_id": cluster_id,
                "avg_distance": 0,
                "sample_count": 0
            })
            continue
        
        # 计算样本到聚类中心的距离
        distances = pairwise_distances(cluster_vectors, [cluster_centers[cluster_id]])
        avg_distance = np.mean(distances)
        intra_distances.append(avg_distance)
        
        cluster_info.append({
            "cluster_id": cluster_id,
            "avg_distance": avg_distance,
            "sample_count": len(cluster_vectors)
        })
    
    # 按类内平均距离从大到小排序
    sorted_indices = np.argsort(intra_distances)[::-1]
    sorted_cluster_info = [cluster_info[i] for i in sorted_indices]
    
    return sorted_cluster_info

def main():
    # 加载模型和数据
    model = load_word2vec_model(r"D:\2\2\model.w2v") #加载词向量模型
    sentences = load_sentence(r"D:\2\2\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化
    
    # 确定聚类数量并执行K-Means
    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)
    
    # 计算并排序类内距离
    cluster_info = calculate_intra_cluster_distances(
        vectors, 
        kmeans.labels_, 
        kmeans.cluster_centers_
    )
    
    # 组织句子按簇分类
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    
    # 按类内距离排序输出结果
    print("\n聚类结果（按类内平均距离从大到小排序）：")
    for info in cluster_info:
        cluster_id = info["cluster_id"]
        print(f"\n簇 {cluster_id}:")
        print(f"类内平均距离: {info['avg_distance']:.4f}")
        print(f"包含样本数: {info['sample_count']}")
        print("代表性句子（最多显示10条）:")
        
        for i in range(min(10, len(sentence_label_dict[cluster_id]))):
            print(sentence_label_dict[cluster_id][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
