#coding: utf-8
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#kmeans根据类内距离排序
def load_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path,encoding='utf-8')as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
            print("句子数量：",len(sentences))
    return sentences

def sentences_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def cosine_distance(vec1,vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))
    return np.sum(vec1*vec2)

def eculid_distance(vec1,vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def main():
    model = load_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences,model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量： ", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_lable_dict = defaultdict(list)
    for sentence,lable in zip(sentences,kmeans.labels_):
        sentence_lable_dict[lable].append(sentence)

    density_dict = defaultdict(list)
    for vector_index, lable in enumerate(kmeans.labels_):
        vector = vectors[vector_index]
        center = kmeans.cluster_centers_[lable]
        distance = cosine_distance(vector, center)
        density_dict[lable].append(distance)

    for lable,distance_list in density_dict.items():
        density_dict[lable] = np.mean(distance_list)
    density_order = sorted(density_dict.items(), key=lambda x : x[1], reverse=True)

    for lable,distance_avg in density_order:
        print('cluster %s, avg distance %f : '%(lable,distance_avg))
        sentences = sentence_lable_dict[lable]
        for i in range(min(10,len(sentences))):
            print(sentences[i].replace(" ",""))
        print("-----")

if __name__== '__main__':
    main()
