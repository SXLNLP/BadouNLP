#!/usr/bin/env python3  
#coding: utf-8


#基于kmeans实现类内距离的排序
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
    
    cluster_centers = kmeans.cluster_centers
    
    distance = []
    for i, vec in enumerate(vectors):
        center = cluster_centers[kmeans.labels_[i]]
        distance = np.linalg.norm(vec - center)
        distances.append(distance)
        
    cluster_sorted_sentences = defaultdict(list)
    for label, sentence, distance in zip(kmeans.labels_, sentence, distances):
        cluster_sorted_sentences[label].append((distance, sentence))
        
    for label in cluster_sorted_sentences:
        cluster_sorted_sentences[label].sort(key = lambda x : x[0])
        
    for label, items in cluster_sorted_sentences.items():
        print(f"cluster {label} (共{len(items)}条):")
        print("最近距离的10条：")
        for i in range(min(10, items)):
            distance, sentence = items[i]
            print(f"{sentence.replace(" ", """)}(距离:{distance:.4f})")

        print("最远距离的10条：")
        for i in range(max(0, len(items) - 10):
            distance, sentence = items[i]
            print(f"{sentence.replace(" ", """)}(距离:{distance:.4f})")
        print("-----------")

if __name__ == "__main__":
    main()
