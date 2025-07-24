#!/usr/bin/env python3
#coding: utf-8

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
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


def calculate_cluster_avg_distance(vectors, labels, centers):
    cluster_distances = defaultdict(list)
    
   
    for vector, label in zip(vectors, labels):
        distance = np.linalg.norm(vector - centers[label])
        cluster_distances[label].append(distance)
    

    cluster_avg_distances = {}
    for label, distances in cluster_distances.items():
        cluster_avg_distances[label] = sum(distances) / len(distances)
    
    return cluster_avg_distances

def main():
    model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v") 
    sentences = load_sentence("titles.txt")  
    vectors = sentences_to_vectors(sentences, model)  

    n_clusters = int(math.sqrt(len(sentences)))  
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters) 
    kmeans.fit(vectors)         

    cluster_avg_distances = calculate_cluster_avg_distance(vectors, kmeans.labels_, kmeans.cluster_centers_)
    sorted_clusters = sorted(cluster_avg_distances.items(), key=lambda x: x[1], reverse=True)
    
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  
        sentence_label_dict[label].append(sentence)         
    
    for label, avg_distance in sorted_clusters:
        sentences_in_cluster = sentence_label_dict[label]
        print(f"cluster {label} (平均距离: {avg_distance:.4f}, 样本数: {len(sentences_in_cluster)}):")
        for i in range(min(10, len(sentences_in_cluster))):  
            print(sentences_in_cluster[i].replace(" ", ""))
        print("-" * 50)

if __name__ == "__main__":
    main()
