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


# 计算类内距离
def calculate_cluster_distances(vectors, kmeans):
    """计算每个聚类的类内距离（平均欧氏距离）"""
    cluster_distances = defaultdict(list)

    # 计算每个样本到其所属聚类中心的距离
    for i, (vector, label) in enumerate(zip(vectors, kmeans.labels_)):
        center = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(vector - center)
        cluster_distances[label].append(distance)

    # 计算每个聚类的平均距离和样本数量
    cluster_metrics = []
    for label, distances in cluster_distances.items():
        avg_distance = sum(distances) / len(distances)
        cluster_size = len(distances)
        cluster_metrics.append({
            'label': label,
            'avg_distance': avg_distance,
            'total_distance': sum(distances),
            'size': cluster_size,
            'sentence_indices': [i for i, lbl in enumerate(kmeans.labels_) if lbl == label]
        })

    return cluster_metrics


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算类内距离
    cluster_metrics = calculate_cluster_distances(vectors, kmeans)

    # 按平均距离排序（从高到低）
    sorted_clusters = sorted(cluster_metrics, key=lambda x: x['avg_distance'], reverse=True)

    # 转换为列表以便按原始顺序访问
    sentence_list = list(sentences)

    # 打印排序后的聚类结果
    print("\n按类内平均距离排序的聚类结果：")
    for i, cluster in enumerate(sorted_clusters):
        print(
            f"排名 #{i + 1} - 聚类 {cluster['label']} (样本数: {cluster['size']}, 平均距离: {cluster['avg_distance']:.4f})")

        # 打印该聚类中距离中心最远的几个样本
        distances = [np.linalg.norm(vectors[idx] - kmeans.cluster_centers_[cluster['label']])
                     for idx in cluster['sentence_indices']]
        sorted_indices = sorted(zip(cluster['sentence_indices'], distances), key=lambda x: x[1], reverse=True)

        print("  代表性样本:")
        for j, (idx, dist) in enumerate(sorted_indices[:3]):  # 打印前3个最具代表性的样本
            original_sentence = sentence_list[idx].replace(" ", "")
            print(f"    {j + 1}. {original_sentence} (距离: {dist:.4f})")
        print("---------")


if __name__ == "__main__":
    main()
