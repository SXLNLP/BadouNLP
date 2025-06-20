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
            if not sentence:
                continue
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
        word_count = 0
        for word in words:
            try:
                vector += model.wv[word]
                word_count += 1
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        if word_count > 0:
            vectors.append(vector / word_count)
        else:
            # 如果句子中所有词都不在词典中，则添加一个零向量
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型, 请确保路径正确
    sentences_set = load_sentence("titles.txt")  # 加载所有标题，得到一个set

    sentences_list = list(sentences_set)
    vectors = sentences_to_vectors(sentences_list, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences_list)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类, n_init='auto' or 10
    kmeans.fit(vectors)  # 进行聚类计算

    # 基于类内距离排序的核心实现

    distances = kmeans.transform(vectors)
    cluster_details = defaultdict(list)

    for i in range(len(sentences_list)):
        label = kmeans.labels_[i]  # 获取第i个句子的簇标签
        sentence = sentences_list[i]  # 获取第i个句子
        distance = distances[i][label]  # 获取第i个句子到其所属簇中心的距离
        cluster_details[label].append((distance, sentence))  # 存入(距离, 句子)元组

    for label in cluster_details:
        cluster_details[label].sort(key=lambda x: x[0])

    print("\n--- 聚类结果（按类内距离排序） ---")
    for label, sorted_items in cluster_details.items():
        print(f"\ncluster {label} (共 {len(sorted_items)} 条):")
        # 打印每个簇中距离中心点最近的前10个句子
        for i in range(min(10, len(sorted_items))):
            dist, sentence = sorted_items[i]
            # 打印时去掉分词的空格，并显示其到中心的距离
            print(f"  (距离: {dist:.4f}) {sentence.replace(' ', '')}")
        print("---------")


if __name__ == "__main__":
    main()
