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
from sklearn.metrics.pairwise import euclidean_distances


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
def calculate_intra_cluster_distance(vectors, labels, cluster_id):
    cluster_vectors = vectors[labels == cluster_id]
    if len(cluster_vectors) <= 1:
        return 0
    distances = euclidean_distances(cluster_vectors)
    total_distance = np.sum(distances)
    num_pairs = len(cluster_vectors) * (len(cluster_vectors) - 1)
    return total_distance / num_pairs


def main():
    model = load_word2vec_model(
        r"D:\BaiduNetdiskDownload\AI架构课程\第五周 词向量\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算每个类别的平均距离
    cluster_distances = []
    for cluster_id in range(n_clusters):
        distance = calculate_intra_cluster_distance(vectors, kmeans.labels_, cluster_id)
        cluster_distances.append((cluster_id, distance))

    # 按平均距离排序
    cluster_distances.sort(key=lambda x: x[1])

    for cluster_id, _ in cluster_distances:
        sentences_in_cluster = sentence_label_dict[cluster_id]
        print("cluster %s :" % cluster_id)
        for i in range(min(10, len(sentences_in_cluster))):  # 随便打印几个，太多了看不过来
            print(sentences_in_cluster[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
