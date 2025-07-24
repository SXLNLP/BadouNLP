#!/usr/bin/env python3  
#coding: utf-8

"""
功課目的：实现基于kmeans结果类内距离的排序
"""

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
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 計算每個聚類的平均距離
    cluster_avg_distances = {}
    for cluster_id in range(n_clusters):
        # 獲取該聚類的所有向量
        cluster_vectors = vectors[kmeans.labels_ == cluster_id]
        if len(cluster_vectors) > 0:
            # 計算該聚類中所有向量到中心的距離
            center = kmeans.cluster_centers_[cluster_id]
            distances = [np.linalg.norm(vector - center) for vector in cluster_vectors]
            avg_distance = np.mean(distances)
            cluster_avg_distances[cluster_id] = avg_distance
        else:
            cluster_avg_distances[cluster_id] = 0
    
    # 按平均距離排序（距離越小越緊密）
    sorted_clusters = sorted(cluster_avg_distances.items(), key=lambda x: x[1])
    
    print("\n=== 聚類平均距離排序結果（距離越小越緊密）===")
    for cluster_id, avg_distance in sorted_clusters:
        print(f"聚類 {cluster_id}: 平均距離 = {avg_distance:.4f}")

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    
    # 按排序後的順序顯示聚類結果
    print("\n=== 聚類結果（按平均距離排序）===")
    for cluster_id, avg_distance in sorted_clusters:
        cluster_sentences = sentence_label_dict[cluster_id]
        print(f"\ncluster {cluster_id} (平均距離: {avg_distance:.4f}):")
        for i in range(min(10, len(cluster_sentences))):  #随便打印几个，太多了看不过来
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

