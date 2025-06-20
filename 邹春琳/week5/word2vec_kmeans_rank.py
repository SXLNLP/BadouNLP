#!/usr/bin/env python3
#coding: utf-8

# 基于训练好的词向量模型进行采用Kmeans算法聚类、并通过计算类间距离排序
import math
import re
import json
import jieba
import numpy as np
from scipy.spatial.distance import pdist
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 训练word2vec模型并保存
def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model

# 输入模型文件路径、加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_stop_words(path):
    stop_words = []
    with open(path, 'r', encoding='utf-8') as f:
        con = f.readlines()
        for line in con:
            stop_words.append(line.replace('\n', ''))
    # print('停用词表长度: ', len(stop_words))
    return stop_words

def load_sentence(path):
    stop_words = load_stop_words(r'hit_stopwords.txt')  # 加载停用词
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            temp = ''
            sentence = line.strip()
            for word in jieba.cut(sentence):
                if word not in stop_words:
                    temp += word
                    temp += ' '
            sentences.add(temp)

            # sentences.append(" ".join(jieba.cut(sentence)))
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
    """
    sentences = []
    with open("corpus.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line))
    train_word2vec_model(sentences, 128)
    """
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    dis_rank_dic = {}
    dis_rank_dic_cos = {}
    for i in range(n_clusters):
        cluster_center = np.array(cluster_centers[i])  # 当前类的中心向量
        cluster_datas = np.array(vectors[labels == i])  # 当前类的所有向量
        # 计算距离：欧几里得、余弦
        cluster_distance = np.zeros_like(cluster_center)
        for cluster_data in cluster_datas:
            distance = np.linalg.norm(cluster_data - cluster_center)
            cluster_distance += distance
        cluster_distance = cluster_distance/len(cluster_datas)
        dis_rank_dic[i] = cluster_distance[0]
        cluster_distance_cos = 1 - pdist(np.vstack([cluster_data, cluster_center]), 'cosine')
        dis_rank_dic_cos[i] = cluster_distance_cos[0]
        # print(f"余弦距离为：{cluster_distance_cos[0]}，欧氏距离为：{cluster_distance[0]}")


    dis_rank_dic = sorted(dis_rank_dic.items(), key=lambda item: item[1])

    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    for i in range(len(dis_rank_dic)):   # 打印类别、类间距离、该类中数据
        distance, label = dis_rank_dic[i][1], dis_rank_dic[i][0]
        print(f"\ncluster : {label}, 类间平均距离(欧式) : {distance}, 类间平均距离(余弦) : {dis_rank_dic_cos[label]}")
        for i in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))


if __name__ == "__main__":
    main()

