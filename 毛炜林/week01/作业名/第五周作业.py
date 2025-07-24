#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法，并实现类内距离计算与类别筛选
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
    print(f"获取句子数量：{len(sentences)}")
    return sentences


# 将文本向量化
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


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    计算两个多维向量的余弦相似度
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0

    similarity = dot_product / (norm_vector1 * norm_vector2)
    return max(min(similarity, 1.0), -1.0)  # 确保值在[-1,1]范围内


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    计算两个多维向量的欧氏距离
    """
    squared_difference = np.sum(np.square(vector1 - vector2))
    distance = np.sqrt(squared_difference)
    return distance


# 定义距离度量方式枚举（移至所有距离函数定义之后）
DISTANCE_METHODS = {
    "cosine": lambda v1, v2: 1 - cosine_similarity(v1, v2),  # 余弦距离 = 1 - 余弦相似度
    "euclidean": euclidean_distance
}


def calculate_intra_cluster_distance(vectors, labels, cluster_centers, method="euclidean"):
    """
    计算每个聚类的类内平均距离

    参数:
        vectors: 所有样本的向量表示
        labels: 每个样本的聚类标签
        cluster_centers: 聚类中心
        method: 距离度量方法，可选'cosine'或'euclidean'

    返回:
        intra_cluster_distances: 每个聚类的类内平均距离
    """
    if method not in DISTANCE_METHODS:
        raise ValueError(f"不支持的距离度量方法: {method}，可选: {list(DISTANCE_METHODS.keys())}")

    distance_func = DISTANCE_METHODS[method]
    intra_cluster_distances = {}

    for cluster_id in np.unique(labels):
        # 获取该聚类中的所有样本
        cluster_samples = vectors[labels == cluster_id]
        # 计算每个样本到聚类中心的距离
        distances = [distance_func(sample, cluster_centers[cluster_id]) for sample in cluster_samples]
        # 计算类内平均距离
        intra_cluster_distances[cluster_id] = np.mean(distances)

    return intra_cluster_distances


def filter_clusters_by_distance(intra_cluster_distances, keep_ratio=0.8):
    """
    根据类内平均距离筛选聚类，保留距离较小的类别

    参数:
        intra_cluster_distances: 每个聚类的类内平均距离
        keep_ratio: 保留的聚类比例，范围(0,1)

    返回:
        kept_clusters: 保留的聚类ID列表
    """
    # 按类内距离从小到大排序
    sorted_clusters = sorted(intra_cluster_distances.items(), key=lambda x: x[1])
    # 计算保留的聚类数量
    n_kept = max(1, int(len(sorted_clusters) * keep_ratio))
    # 保留距离较小的聚类
    kept_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:n_kept]]
    print(f"原始聚类数量: {len(intra_cluster_distances)}, 保留聚类数量: {n_kept}")
    return kept_clusters


def main():
    model = load_word2vec_model(
        r"F:\1\八斗精品班\第五周\week5+词向量及文本向量\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print(f"指定聚类数量：{n_clusters}")
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    cluster_centers = kmeans.cluster_centers_

    # 计算类内平均距离 (使用欧式距离为例，可改为'cosine')
    intra_distances = calculate_intra_cluster_distance(
        vectors, kmeans.labels_, cluster_centers, method="euclidean"
    )

    # 按类内距离排序并筛选聚类
    kept_clusters = filter_clusters_by_distance(intra_distances, keep_ratio=0.8)

    # 整理每个保留聚类中的句子
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        if label in kept_clusters:  # 只保留符合条件的聚类
            sentence_label_dict[label].append(sentence)

    # 输出每个聚类的信息和句子示例
    for label in sorted(kept_clusters):
        cluster_sentences = sentence_label_dict[label]
        avg_distance = intra_distances[label]
        print(f"\n聚类 {label} (类内平均距离: {avg_distance:.4f}) - 句子数量: {len(cluster_sentences)}")
        for i in range(min(5, len(cluster_sentences))):  # 打印前5个句子
            print(cluster_sentences[i].replace(" ", ""))


if __name__ == "__main__":
    main()
