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


#作业部分
#计算每个聚类的平均中心距离并排序
def calculate_avg_distance(sentence_label_dict, vectors, kmeans):
    distances = []
    for label, sentences in sentence_label_dict.items():
        cluster_vectors = [vectors[i] for i, sent in enumerate(sentences) if sent in sentences]
        # 获取该聚类的中心点向量
        center = kmeans.cluster_centers_[label]
        total_distance = 0
        for vector in cluster_vectors:
            # 使用欧氏距离计算向量到中心点的距离
            total_distance += np.linalg.norm(vector - center)
        # 计算平均距离
        avg_distance = total_distance / len(cluster_vectors)
        distances.append((label, avg_distance))
    # 按照平均距离从短到长排序
    distances.sort(key=lambda x: x[1])
    return distances


def main():
    model = load_word2vec_model(r"H:\八斗网课\第五周 词向量及文本向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算



    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # 计算平均中心距离并排序
    sorted_distances = calculate_avg_distance(sentence_label_dict, vectors, kmeans)

    # 按照排序后的顺序输出分类结果
    for label, avg_distance in sorted_distances:
        print(f"cluster {label} (Average Distance: {avg_distance:.4f}):")
        cluster_sentences = sentence_label_dict[label]
        for i in range(min(10, len(cluster_sentences))):  #随便打印几个，太多了看不过来
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")

        # for label, sentences in sentence_label_dict.items():
        #     print("cluster %s :" % label)
        #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
        #         print(sentences[i].replace(" ", ""))
        #     print("---------")

if __name__ == "__main__":
    main()

