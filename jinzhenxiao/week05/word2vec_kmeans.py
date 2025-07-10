#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
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


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        sentence_label_dict[label].append(sentences[i])
        vector_label_dict[label].append(vectors[i])

    cluster_distances = {}
    for label, vecs in vector_label_dict.items():
        centroid = kmeans.cluster_centers_[label]
        distance_sum = sum(np.linalg.norm(vec - centroid) for vec in vecs)
        avg_distance = distance_sum / len(vecs)
        cluster_distances[label] = avg_distance

    # 排序
    sorted_clusters = sorted(cluster_distances.items(), key=lambda item: item[1])
    for label, avg_dist in sorted_clusters:
        sentences_in_cluster = sentence_label_dict[label]
        print(f"\nCluster {label} (簇內平均距离: {avg_dist:.4f}):")
        for i in range(min(10, len(sentences_in_cluster))):
            print(f"  - {sentences_in_cluster[i].replace(' ', '')}")
        print("-----------------------------------------")


if __name__ == "__main__":
    main()

