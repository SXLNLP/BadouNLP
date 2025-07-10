#!/usr/bin/env python3  
#coding: utf-8

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

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model("model.w2v")  # 修改为你的模型路径
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)

    # 计算每个聚类的类内平均距离
    cluster_distances = []
    for i in range(n_clusters):
        # 获取当前聚类的所有向量
        cluster_vectors = vectors[kmeans.labels_ == i]
        if len(cluster_vectors) > 0:
            # 计算到聚类中心的距离
            distances = np.linalg.norm(cluster_vectors - kmeans.cluster_centers_[i], axis=1)
            avg_distance = np.mean(distances)
            cluster_distances.append((i, avg_distance))
        else:
            cluster_distances.append((i, 0))
    
    # 按平均距离排序（升序：距离越小表示类内越紧密）
    sorted_clusters = sorted(cluster_distances, key=lambda x: x[1])
    
    # 组织句子到聚类
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    
    # 按排序结果输出聚类
    for cluster_id, avg_dist in sorted_clusters:
        cluster_sentences = sentence_label_dict[cluster_id]
        print(f"cluster {cluster_id} (平均距离: {avg_dist:.4f}):")
        for i in range(min(10, len(cluster_sentences))):
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()