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

#计算某一组所有样本点的中心点，即一组向量的质心(所有向量纵向，即按行的方向取平均数) cuibaoxiong新写
def get_center_vector(vectors):
    return np.mean(vectors, axis=0)

#计算两点之间的距离（欧式距离） cuibaoxiong新写
def get_distance(vector1, vector2):
    vector_diff = [np.power(i, 2) for i in vector1 - vector2] #两个向量先做减法，再计算平方，得到新的向量
    distance = np.power(np.sum(vector_diff), 0.5) #将得到的新向量，求和再开方
    return distance

#计算某一组所有样本点到质心的平均距离， cuibaoxiong新写
def get_distance_avg(vectors, center):
    if(len(vectors) == 0):
        return 0

    sum_distiance = 0
    for vector in vectors:
        sum_distiance += get_distance(vector, center)

    return sum_distiance / len(vectors)

def main():
    model = load_word2vec_model(r"gensim_Word2Vec_model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    print(f"sentences数量{len(sentences)}，sentences：{sentences}\n")
    print(f"kmeans.labels数量：{len(kmeans.labels_)}，{kmeans.labels_}\n", )

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #print(f"label：{label}")
        #print(f"sentence：{sentence}")
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)

        """ 这是week5的作业代码   -----------start--------  """
        #当前这一组句子的向量
        vectors_label = sentences_to_vectors(sentences, model)
        #计算这一组向量的中心点向量
        vector_center = get_center_vector(vectors_label)
        #计算当前这一组所有句子的向量到中心点向量的平均距离（欧式距离）
        distance_label_avg = get_distance_avg(vectors_label, vector_center)
        #打印出平均距离
        print(f"平均距离：{distance_label_avg}\n")
        """ 这是week5的作业部分代码   -----------end--------  """

        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))

        print("------------------------------------")

if __name__ == "__main__":
    main()

