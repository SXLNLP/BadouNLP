import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from numpy.f2py.crackfortran import word_pattern
from sklearn.cluster import KMeans

import os
os.environ['OMP_NUM_THREADS'] = '8'
def load_word2vec_model(path):
    model =Word2Vec.load(path)
    return model

# 加载文本
def load_sentence(path):
    sentences = set()
    with open(path,encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.lcut(sentence)))
    print(f"句子数量:{len(sentences)}")
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        vector = np.zeros(model.vector_size)
        words = sentence.split()
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(words))
    return np.array(vectors)

# 计算类内平均距离
def average_distance(center,vct):
    distance = []
    for vector in vct:
        s = 0
        for i in range(len(vector)):
            s += pow(vector[i]-center[i],2)
        s = math.sqrt(s)
        distance.append(s)
    return np.mean(distance)
def main():
    model = load_word2vec_model("model.w2v") #加载模型
    sentences = load_sentence("titles.txt") #加载文本
    # 转换文本向量
    vectors = sentences_to_vectors(sentences,model)
    # 簇的数量
    n_clusters = int(math.sqrt(len(sentences)))
    # 定义kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    # 聚类计算
    kmeans.fit(vectors)
    # 输出每个文本的聚类
    label_sentences_dict = defaultdict(list)
    label_vectors_dict = defaultdict(list)
    avg_dict = {}
    for label,vector,sentence in zip(kmeans.labels_,vectors,sentences):
        label_sentences_dict[label].append(sentence)
        label_vectors_dict[label].append(vector)
    # 计算每个类的类内距离，并放在avg_dict
    for label,vct in label_vectors_dict.items():
        avg_dis = average_distance(kmeans.cluster_centers_[label],vct)
        avg_dict[label] = avg_dis
    avg_dict = sorted(avg_dict.items(),key = lambda x:x[1])
    print(avg_dict)
    for label,avg_dis in avg_dict:
        print(f"cluster：{label}，类内平均距离：{avg_dis}")
        for i in range(min(5,len(label_sentences_dict[label]))):
            print(label_sentences_dict[label][i].replace(" ",""))
        print("-------------")

if  __name__ == "__main__":
    main()
