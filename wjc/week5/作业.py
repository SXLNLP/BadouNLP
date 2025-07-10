#!/usr/bin/env python3  
#coding: utf-8

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
    model = load_word2vec_model("model.w2v")  #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)  #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)  #进行聚类计算
    distant_list = [np.linalg.norm(vec - kmeans.cluster_centers_[label]) for vec, label in zip(vectors, kmeans.labels_)]
    sentences_distant_dict = defaultdict(list)
    for distant, label, sentence in zip(distant_list, kmeans.labels_, sentences):
        sentences_distant_dict[label].append((distant, sentence))
    result = defaultdict(list)
    for label in sentences_distant_dict:
        avg_distant = np.mean([distant for distant, _ in sentences_distant_dict[label]])
        sorted_sentences = sorted([(d, s) for d, s in sentences_distant_dict[label] if d < avg_distant], key=lambda x: x[0], reverse=False)
        result[label].append([avg_distant])
        result[label].append([(d, s.replace(" ", "")) for d, s in sorted_sentences])
    for label in result:
        result_dict = result[label]
        for index, sentence in enumerate(result_dict):
            if index == 0:
                print(f"cluster: {label};avg: {sentence[0]}")
            else:
                for i, s in enumerate(sentence):
                    print(f"{i+1}.{s[1]}---{s[0]}")
        print("-------------------")


if __name__ == "__main__":
    main()
