import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

from test_model import sentences_to_vectors


#加载训练好的模型
def load_word2vec_model(model_path):
    model = Word2Vec.load(model_path)
    return model

#加载语料
def load_sentence(sentence_path):
    sentences = set()
    with open(sentence_path, "r", encoding="utf-8") as f:
        for line in f:
           sentence = line.strip()
           sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量，sentence num: ", len(sentences))
    my_list = list(sentences)
    for i in range(10):
        print(my_list[i].replace(" ", ""))
        print("-----------")
    return sentences

#文本向量化
def sentence_to_vector(sentences, model):
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

#计算向量到中心点距离
def get_average_distance(center, vectors):
    # 计算每个向量到点的欧氏距离
    distances = np.linalg.norm(vectors - center, axis=1)
    # 计算所有距离的平均值
    average_distance = np.mean(distances)
    return average_distance

def main():
    model = load_word2vec_model("model.w2v") #加载模型
    sentences = load_sentence("titles.txt") #加载标题
    vectors = sentences_to_vectors(sentences, model) #标题向量化

    n_clusters = int(math.sqrt(len(sentences))) #计算聚类数量
    kmeans = KMeans(n_clusters=n_clusters, random_state=0) #kmeans聚类计算
    kmeans.fit(vectors)

    #字典，按照标签，将句子和向量分组
    sentence_label_dict = defaultdict(list)
    vectors_label_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):
        sentence_label_dict[label].append(sentence) #标题分组
        vectors_label_dict[label].append(vector) #向量分组

    #字典，计算每个标签组内所有向量到中心点的平均值距离
    avg_distance = {}
    for label, vectors in vectors_label_dict.items():
        average = get_average_distance(kmeans.cluster_centers_[label], vectors)
        avg_distance[label] = average
    #根据距离排序
    average_dis = sorted(avg_distance.items(), key=lambda x: x[1])

    for label, distance in average_dis:
        print(f"cluster：{label}，类内平均距离：{distance}")
        for i in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("-----------------------")

if __name__ == '__main__':
    main()
