# 基于训练好的词向量模型使用Kmeans算法进行聚类

import numpy as np
import math
import jieba
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from collections import defaultdict

# 输入模型文件路径


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载训练好的模型


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print(f"获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好的，使用空格分开
        vector = np.zeros(model.vector_size)
        # 将所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，使用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算每个聚类簇内部的平均距离
def calculate_inner_cluster_distance(vectors, labels, centers):
    n_clusters = centers.shape[0]
    cluster_distances = []

    for i in range(n_clusters):
        # 找到第i个簇的所有向量
        cluster_points = vectors[labels == i]

        if len(cluster_points) == 0:
            cluster_distances.append(0)
            continue

        # 计算这些向量到该簇中心的欧式距离
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)

        # 平均距离
        avg_distance = np.mean(distances)
        cluster_distances.append(avg_distance)

        print(f"簇{i}的样本数为：{len(cluster_points)}, 平均距离：{avg_distance:.4f}")
    return cluster_distances


# 使用柱状图表示每个簇的平均距离
def cluster_histogram(avg_distances):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_distances)),avg_distances)
    plt.xlabel("Cluster")
    plt.ylabel("Average distance")
    plt.show()


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")   # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print(f"指定聚类数量：{n_clusters}")
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)         # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print(f"cluster:{label}")
        for i in range(min(10, len(sentences))):  # 在指定范围内打印几个聚类及其内容
            print(sentences[i].replace(" ", ""))
        print("---------------")

    # 计算每个簇的内部平均距离
    inner_distances = calculate_inner_cluster_distance(vectors, kmeans.labels_, kmeans.cluster_centers_)

    # 使用柱状图可视化每个簇的平均距离
    cluster_histogram(inner_distances)


if __name__ == "__main__":
    main()
