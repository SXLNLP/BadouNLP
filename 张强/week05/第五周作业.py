#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
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
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
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

    centers = np.array(kmeans.cluster_centers_)
    print(centers)
    sentence_label_vectors = defaultdict(list)
    sequence = {}
    for vector, label in zip(vectors, kmeans.labels_):  # 取出向量和标签
        sentence_label_vectors[label].append(vector)
    for label in range(n_clusters):
        vectors_in_cluster = np.array(sentence_label_vectors[label])
        center = centers[label]
        distances = np.linalg.norm(vectors_in_cluster - center, axis=1)
        print(f"簇 {label}:")
        print(f"  样本数: {len(vectors_in_cluster)}")
        print(f"  平均距离: {np.mean(distances):.4f}")
        print(f"  最大距离: {np.max(distances):.4f}")
        sequence[label] = np.mean(distances)
    print(sequence)
    print(sorted(sequence.items(), key=lambda item: item[1], reverse=True))
