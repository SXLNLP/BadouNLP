#!/usr/bin/env python3
#coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型
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

# 将文本向量化
def sentences_to_vectors(sentences, model):
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


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    # ========== 按距离排序输出 ==========
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    cluster_dict = defaultdict(list)

    for sentence, label, vector in zip(sentences, labels, vectors):
        center = centers[label]
        distance = np.linalg.norm(vector - center)  # 计算欧氏距离
        cluster_dict[label].append((distance, sentence))

    print("\n\n===== 聚类结果（按类内距离排序） =====")
    for label, items in cluster_dict.items():
        print(f"\nCluster {label}:")
        # 按距离排序
        sorted_items = sorted(items, key=lambda x: x[0])
        for distance, sentence in sorted_items[:10]:  # 只展示前10个
            print(f"[距离: {distance:.4f}] {sentence.replace(' ', '')}")
        print("-" * 30)

if __name__ == "__main__":
    main()
