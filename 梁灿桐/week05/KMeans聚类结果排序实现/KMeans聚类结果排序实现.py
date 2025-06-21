#!/usr/bin/env python3
# coding: utf-8

import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

'''
实现基于kmeans结果类内距离的排序
'''

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
    model = load_word2vec_model(r"model.w2v")
    sentences = list(load_sentence("titles.txt"))  # 转换为列表以保持顺序
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init=10)  # 添加n_init参数避免警告
    kmeans.fit(vectors)

    # 获取簇中心
    cluster_centers = kmeans.cluster_centers_

    # 存储每个簇的(句子, 距离)列表
    cluster_sentences = defaultdict(list)

    # 计算每个句子到所属簇中心的距离
    for idx, label in enumerate(kmeans.labels_):
        center = cluster_centers[label]
        distance = np.linalg.norm(vectors[idx] - center)  # 欧氏距离
        cluster_sentences[label].append((sentences[idx], distance))

    # 对每个簇内的句子按距离排序（升序）
    for label in cluster_sentences:
        cluster_sentences[label].sort(key=lambda x: x[1])

    # 输出每个簇的排序结果
    for label, sentences_dist in cluster_sentences.items():
        print(f"cluster {label} (中心距离排序):")
        for i, (sentence, dist) in enumerate(sentences_dist[:10]):  # 取前10个
            print(f"{sentence.replace(' ', '')} (距离: {dist:.4f})")
        print("---------")


if __name__ == "__main__":
    main()
