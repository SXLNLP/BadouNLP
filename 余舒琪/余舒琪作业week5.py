# 基于KMeans结果类内距离的排序

import numpy as np
import jieba
import math
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载文本
def load_sentence(path):
    sentences = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print(f"获取句子数量：{len(sentences)}")
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        vector = np.zeros(model.vector_size)
        words = sentence.split()
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算类内平均距离
def average_distance(center, vectors):
    distance = []
    for vector in vectors:
        sum = 0
        for i in range(len(vector)):
            sum += pow(vector[i] - center[i], 2)
        sum = pow(sum, 0.5)
        distance.append(sum)
    distance = np.mean(distance)
    return distance

def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)
    n_clusters = int(math.sqrt(len(sentences)))
    print(f"指定聚类数量：{n_clusters}")
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentences_label_dict = defaultdict(list)
    vectors_label_dict = defaultdict(list)
    average_dis = {}
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        sentences_label_dict[label].append(sentence)
        vectors_label_dict[label].append(vector)
    for label, vectors in vectors_label_dict.items():
        average = average_distance(kmeans.cluster_centers_[label], vectors)
        average_dis[label] = average
    average_dis = sorted(average_dis.items(), key=lambda x: x[1])
    print(average_dis)
    for label, distance in average_dis:
        print(f"cluster：{label}，类内平均距离：{distance}")
        for i in range(min(10, len(sentences_label_dict[label]))):
            print(sentences_label_dict[label][i].replace(" ", ""))
        print("-----------------------")

if __name__ == "__main__":
    main()
