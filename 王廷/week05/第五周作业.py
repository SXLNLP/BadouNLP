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
    sentences = []  # 使用列表而不是集合
    seen = set()    # 用于去重的集合
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            # 使用jieba分词
            seg_sentence = " ".join(jieba.cut(sentence))
            # 去重处理
            if seg_sentence not in seen:
                seen.add(seg_sentence)
                sentences.append(seg_sentence)
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
        # 避免除以零的情况
        if len(words) > 0:
            vector /= len(words)
        vectors.append(vector)
    return np.array(vectors)


def main():
    # 模型和文件路径
    model_path = r"D:\BaiduNetdiskDownload\第五周\八斗精品班\week5 词向量及文本向量（Lkq48CiC）\week5 词向量及文本向量\model.w2v"
    titles_path = r"D:\BaiduNetdiskDownload\第五周\八斗精品班\week5 词向量及文本向量（Lkq48CiC）\week5 词向量及文本向量\titles.txt"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    if not os.path.exists(titles_path):
        print(f"错误：找不到标题文件 {titles_path}")
        return
    
    # 加载模型和句子
    model = load_word2vec_model(model_path)  # 加载词向量模型
    sentences = load_sentence(titles_path)   # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    # 计算聚类数量（取平方根，但至少为1）
    n_clusters = max(1, int(math.sqrt(len(sentences))))
    print(f"句子数量: {len(sentences)}, 指定聚类数量: {n_clusters}")
    
    # 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 获取聚类中心和标签
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # 存储每个聚类的句子及其到中心的距离
    cluster_sentences = defaultdict(list)
    
    # 计算每个句子到所属聚类中心的距离
    for i, label in enumerate(labels):
        sentence = sentences[i]
        # 计算欧氏距离
        distance = np.linalg.norm(vectors[i] - cluster_centers[label])
        cluster_sentences[label].append((sentence, distance, i))  # 添加索引以便后续分析
    
    # 对每个聚类内的句子按距离排序（从小到大）
    for label in cluster_sentences:
        cluster_sentences[label].sort(key=lambda x: x[1])
    
    # 输出排序后的聚类结果
    for label, sentences_dist in sorted(cluster_sentences.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n===== 聚类 {label} (样本数: {len(sentences_dist)}) =====")
        print(f"中心点位置: {cluster_centers[label][:5]}... (显示前5维)")
        
        # 输出每个聚类的前10个句子（距离最近的10个）
        print("\n最具代表性的句子（距离中心最近）:")
        for i in range(min(10, len(sentences_dist))):
            # 去掉分词空格输出原句
            print(f"{i+1}. {sentences_dist[i][0].replace(' ', '')} (距离: {sentences_dist[i][1]:.4f})")
        
        # 输出距离中心最远的几个句子
        if len(sentences_dist) > 10:
            print("\n距离中心最远的句子:")
            for i in range(1, min(4, len(sentences_dist))):
                idx = -i
                print(f"{i}. {sentences_dist[idx][0].replace(' ', '')} (距离: {sentences_dist[idx][1]:.4f})")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
