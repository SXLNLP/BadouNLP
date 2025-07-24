#!/usr/bin/env python3  
#coding: utf-8
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
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


def main():
    model = load_word2vec_model(r"/Users/chenayu/Desktop/第五周/八斗精品班/week5 词向量及文本向量/model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    sentences_list = list(sentences)  # 转换为列表保持顺序
    vectors = sentences_to_vectors(sentences_list, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences_list)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算每个样本到所属簇中心的距离
    distances = np.zeros(len(vectors))
    for i in range(len(vectors)):
        cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
        distances[i] = np.linalg.norm(vectors[i] - cluster_center)

    # 按聚类标签和距离组织数据
    cluster_data = defaultdict(list)
    for sentence, label, distance, vector in zip(sentences_list, kmeans.labels_, distances, vectors):
        cluster_data[label].append((sentence, distance, vector))

    # 对每个簇内的句子按距离排序（从近到远）
    for label, items in cluster_data.items():
        sorted_items = sorted(items, key=lambda x: x[1])  # 按距离排序
        print(f"cluster {label} (中心距离排序):")
        print(f"簇内样本数: {len(sorted_items)}")
        
        # 打印距离中心最近的5个句子
        print("【最近距离】:")
        for i in range(min(5, len(sorted_items))):
            print(f"{sorted_items[i][0].replace(' ', '')} (距离: {sorted_items[i][1]:.4f})")
        
        # 打印距离中心最远的5个句子
        if len(sorted_items) > 5:
            print("【最远距离】:")
            for i in range(max(0, len(sorted_items)-5), len(sorted_items)):
                print(f"{sorted_items[i][0].replace(' ', '')} (距离: {sorted_items[i][1]:.4f})")
        
        # 打印簇统计信息
        avg_distance = np.mean([dist for _, dist, _ in sorted_items])
        print(f"平均距离: {avg_distance:.4f} | 最小距离: {sorted_items[0][1]:.4f} | 最大距离: {sorted_items[-1][1]:.4f}")
        print("---------", "\n")

if __name__ == "__main__":
    main()
