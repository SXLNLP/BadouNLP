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


# 计算类内平均距离
def calculate_intra_cluster_distance(vectors, cluster_indices):
    """
    计算簇内样本点之间的平均距离
    参数:
    vectors: 所有样本的向量表示
    cluster_indices: 该簇内样本的索引列表
    返回:
    类内平均距离
    """
    if len(cluster_indices) <= 1:
        return 0.0

    cluster_vectors = vectors[cluster_indices]
    distances = []

    # 计算簇内每对样本之间的欧氏距离
    for i in range(len(cluster_vectors)):
        for j in range(i + 1, len(cluster_vectors)):
            distance = np.linalg.norm(cluster_vectors[i] - cluster_vectors[j])#计算两个向量之间的欧氏距离
            distances.append(distance)

    # 返回平均距离
    return np.mean(distances) if distances else 0.0


def main():
    model = load_word2vec_model(r"E:\PycharmProjects\NLPtask\week5   词向量和文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    cluster_indices_dict = defaultdict(list)

    # 收集每个簇的句子和对应的索引
    for i, (sentence, label) in enumerate(zip(sentences, kmeans.labels_)):
        sentence_label_dict[label].append(sentence)
        cluster_indices_dict[label].append(i)

    # 计算每个簇的类内平均距离并存储
    cluster_distances = []
    for label, indices in cluster_indices_dict.items():
        avg_distance = calculate_intra_cluster_distance(vectors, indices)
        cluster_distances.append((label, avg_distance, len(indices)))

    # 按照类内平均距离从小到大排序
    cluster_distances.sort(key=lambda x: x[1])

    # 输出排序后的聚类结果
    print("\n聚类结果（按类内平均距离从小到大排序）：")
    for label, avg_distance, count in cluster_distances:
        print(f"\ncluster {label} (类内平均距离: {avg_distance:.4f}, 包含句子数: {count}):")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  # 打印前10个句子
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
