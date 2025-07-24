import math

import numpy as np
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity


def load_sentences(filename):
    sentences = set()
    with open(filename) as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    return sentences


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r'model.w2v')
    sentences = load_sentences('titles.txt')
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vectors)
    print("---------")

    # 计算每个聚类的内聚集度（已自动平均）
    labels = kmeans.fit_predict(vectors)
    intra_similarities = calculate_cluster_intra_similarity(vectors, labels)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for label, sentences in sentence_label_dict.items():
        print("cluster %s  label %s :" % (label, intra_similarities[label]))
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


def calculate_cluster_intra_similarity(vectors, labels):
    """
    计算每个聚类的内聚集度

    参数:
    vectors: 样本向量矩阵，形状为 (n_samples, n_features)
    labels: 聚类标签数组，形状为 (n_samples,)

    返回:
    dict: 每个聚类标签对应的内聚集度
    """
    # 按聚类标签分组样本索引
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    # 计算每个聚类的内聚集度
    intra_similarities = {}

    for label, indices in cluster_indices.items():
        # 获取当前聚类的所有样本向量
        cluster_vectors = vectors[indices]
        n_samples = len(cluster_vectors)

        if n_samples <= 1:
            # 只有一个样本的聚类，相似度设为1或NaN
            intra_similarities[label] = np.nan
            continue

        # 计算余弦相似度矩阵
        sim_matrix = cosine_similarity(cluster_vectors)

        # 排除对角线（自己与自己的相似度为1）
        total_similarity = np.sum(sim_matrix) - n_samples  # 减去对角线元素之和

        # 计算平均相似度（n*(n-1)个非对角线元素）
        avg_similarity = total_similarity / (n_samples * (n_samples - 1))
        intra_similarities[label] = avg_similarity

    return intra_similarities


if __name__ == "__main__":
    main()
