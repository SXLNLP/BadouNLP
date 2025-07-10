#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
import os
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
    model = load_word2vec_model(r"C:\Users\cj783\Desktop\AI算法工程师\week5 词向量及文本向量\model.w2v") #加载词向量模型
    
    current_dir = os.path.dirname(__file__)  # 获取当前脚本的路径
    file_path = os.path.join(current_dir, "titles.txt")
    sentences = load_sentence(file_path)  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    sent_list = list(sentences)

    cluster_sentence_dist = defaultdict(list)
    for idx, (sentence, label) in enumerate(zip(sent_list, labels)):
        vec = vectors[idx]
        center = centers[label]
        dist = np.linalg.norm(vec - center)  # 欧几里得距离
        cluster_sentence_dist[label].append((sentence, dist))


    for label, sent_dist_list in cluster_sentence_dist.items():
        print(f"\nCluster {label}:")
        sorted_sents = sorted(sent_dist_list, key=lambda x: x[1])
        for sentence, dist in sorted_sents[:10]:  # 每类最多输出10条
            print(f"{sentence.replace(' ', '')}（距离：{dist:.4f}）")

if __name__ == "__main__":
    main()

