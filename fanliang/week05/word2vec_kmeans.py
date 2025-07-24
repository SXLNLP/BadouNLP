#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    model = load_word2vec_model("/Users/fanliangliang/八斗/badoucode/week5 词向量及文本向量/model.w2v") #加载词向量模型
    sentences = load_sentence("/Users/fanliangliang/八斗/badoucode/week5 词向量及文本向量/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    '''
    defaultdict(list) 创建了一个特殊的字典，这个字典的默认值是空列表 []。
    作用：
    当你访问一个不存在的 key 时，比如 sentence_label_dict[3]，它会自动创建一个空列表作为初始值，而不会报错。
    这样你可以直接用 sentence_label_dict[label].append(sentence) 给某个标签下的句子分组，无需先判断 key 是否存在。
    '''
    sentence_label_dict = defaultdict(list)
    #lambda:[0] 创建了一个匿名函数，返回一个包含单个元素的列表，初始值为0，这个参数必须是函数，不能是别的类型
    means_dict = defaultdict(lambda:[0])
    counts = np.bincount(kmeans.labels_)
    for sentence, label  ,vectoritem in zip(sentences, kmeans.labels_,vectors):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        dist = np.linalg.norm(kmeans.cluster_centers_[label] - vectoritem)
        means_dict[label][0]+=(dist/counts[label])
    '''
    key=lambda x: x[1][0]
    x 是每个键值对元组
    x[1] 是值（一个列表）
    x[1][0] 是这个列表的第一个元素
    排序会根据这个值进行
    '''
    means_sort_dict = sorted(means_dict.items(), key=lambda x: x[1][0])
    #将means_sort_dict转为tensor，并进行softmax
    means_tensor = torch.stack([torch.tensor(value[1][0]) for value in means_sort_dict])
    softmeans = torch.softmax(means_tensor, dim=0)
    #取出means_sort_dict的key
    sort_keys = [item[0] for item in means_sort_dict]
    #将sentence_label_dict按sort_keys的顺序排序
    sentence_sorted_items = sorted(sentence_label_dict.items(),key=lambda item: sort_keys.index(item[0]) if item[0] in sort_keys else len(sort_keys))
    #打印排序后的结果
    for label, sentences in sentence_sorted_items:
        print("cluster %s :" % label)
        print("itemcount：", counts[label])
        print("meandistence：", means_dict[label])
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

