#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法，并实现基于类内距离的排序
import math
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
    return list(sentences)  # 转换为列表以保持顺序


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


# 计算类内距离并排序
def sort_by_intra_distance(cluster_dict, vectors, centroids):
    sorted_clusters = {}

    for label, indices in cluster_dict.items():
        # 获取当前聚类的中心点
        centroid = centroids[label]

        # 获取当前聚类的所有向量
        cluster_vectors = vectors[indices]

        # 计算每个向量到中心点的距离
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

        # 将索引和距离组合，并按距离排序
        sorted_indices = sorted(zip(indices, distances), key=lambda x: x[1])

        # 存储排序后的结果
        sorted_clusters[label] = sorted_indices

    return sorted_clusters


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)

    # 使用KMeans++初始化方法以获得更好的聚类结果
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(vectors)  # 进行聚类计算

    # 获取聚类中心
    centroids = kmeans.cluster_centers_

    # 创建一个字典来存储每个聚类的索引
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(kmeans.labels_):
        cluster_dict[label].append(idx)

    # 对每个聚类按类内距离排序
    sorted_clusters = sort_by_intra_distance(cluster_dict, vectors, centroids)

    # 输出聚类结果
    for label, sorted_items in sorted_clusters.items():
        print(f"\n===== 聚类 {label} (共 {len(sorted_items)} 个样本) =====")
        print(f"聚类中心坐标: {centroids[label][:5]}...")  # 只显示前5维

        # 计算并显示类内平均距离
        avg_distance = np.mean([dist for _, dist in sorted_items])
        print(f"类内平均距离: {avg_distance:.4f}")

        print("\n距离聚类中心最近的10个句子:")
        for i, (idx, distance) in enumerate(sorted_items[:10]):
            print(f"{i + 1}. [距离:{distance:.4f}] {sentences[idx].replace(' ', '')}")

        print("\n距离聚类中心最远的5个句子:")
        for i, (idx, distance) in enumerate(sorted_items[-5:]):
            print(f"{len(sorted_items) - i}. [距离:{distance:.4f}] {sentences[idx].replace(' ', '')}")

        print("---------")


if __name__ == "__main__":
    main()
