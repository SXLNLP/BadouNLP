# coding: utf-8

"""
基于训练好的词向量模型进行KMeans聚类，并按照每个簇内样本到中心点的距离进行排序。
"""

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    """
    加载本地训练好的 Word2Vec 模型
    """
    model = Word2Vec.load(path)
    return model


def load_sentences(path):
    """
    加载文本文件，进行分词并去重
    """
    sentences = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            # 使用jieba进行分词，空格连接
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return list(sentences)  # 保持顺序，方便与向量一一对应


def sentence_to_vector(sentence, model):
    """
    将单个句子向量化（取词向量的平均）
    """
    words = sentence.split()
    vector = np.zeros(model.vector_size)
    valid_words = 0
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
            valid_words += 1
    if valid_words == 0:
        return vector  # 返回全零向量
    return vector / valid_words


def sentences_to_vectors(sentences, model):
    """
    将所有句子转为向量
    """
    return np.array([sentence_to_vector(s, model) for s in sentences])


def cluster_and_sort(sentences, vectors, n_clusters):
    """
    使用KMeans聚类并根据类内距离排序
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    # 计算每个样本到其聚类中心的欧氏距离
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[labels], axis=1)

    # 组织数据：{label: [(distance, sentence), ...]}
    cluster_map = defaultdict(list)
    for label, sentence, distance in zip(labels, sentences, distances):
        cluster_map[label].append((distance, sentence))

    # 对每个簇内的句子，先计算平均距离，再筛选出“小于平均值”的样本，并按距离升序排序
    result = defaultdict(list)
    for label, items in cluster_map.items():
        avg_dist = np.mean([dist for dist, _ in items])
        # 距离小于平均值的样本（越靠近中心越典型）
        filtered_sorted = sorted(
            [(d, s.replace(" ", "")) for d, s in items if d < avg_dist],
            key=lambda x: x[0]
        )
        result[label] = {
            "avg_distance": avg_dist,
            "sorted_sentences": filtered_sorted
        }

    return result


def print_cluster_result(result):
    """
    输出聚类结果
    """
    for label in sorted(result.keys()):
        cluster = result[label]
        print(f"Cluster: {label} | Avg Distance: {cluster['avg_distance']:.4f}")
        for i, (dist, sentence) in enumerate(cluster["sorted_sentences"], start=1):
            print(f"{i}. {sentence} --- {dist:.4f}")
        print("---------------------------")


def main():
    model_path = "model.w2v"
    text_path = "titles.txt"

    # 加载模型和数据
    model = load_word2vec_model(model_path)
    sentences = load_sentences(text_path)
    vectors = sentences_to_vectors(sentences, model)

    # 自动估算簇数（√N法则）
    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)

    # 聚类 & 排序
    result = cluster_and_sort(sentences, vectors, n_clusters)

    # 输出
    print_cluster_result(result)


if __name__ == "__main__":
    main()
