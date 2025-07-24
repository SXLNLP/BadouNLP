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
from numpy.linalg import norm


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(model_path):
    model = Word2Vec.load(model_path)
    return model


def load_sentence(sentence_path):
    sentences = set()
    with open(sentence_path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本转向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence已分好，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # unk使用0向量表达
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model('model.w2v')  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", clusters)
    kmeans = KMeans(clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    sentence_vector_dict = {
        sentence: vector for sentence, vector in zip(sentences, vectors)
    }
    # print(sentence_vector_dict['海渔 广场 共创 城市 商务 价值 新 中心 论坛 （ 4 ）'])
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        vector = sentence_vector_dict[sentence]
        sentence_label_dict[label].append((sentence, vector))
    cluster_avg_distance_list = []

    # 遍历每个簇，计算平均距离
    for label, sentence_vector_list in sentence_label_dict.items():
        vectors = [vec for _, vec in sentence_vector_list]
        total_dist = 0
        count = 0

        # 计算所有两两句子的距离
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += norm(vectors[i] - vectors[j])
                count += 1

        if count == 0:
            avg_dist = 0
        else:
            avg_dist = total_dist / count

        cluster_avg_distance_list.append((label, avg_dist, sentence_vector_list))

    # 从最聚集的类到最松散的
    cluster_avg_distance_list.sort(key=lambda x: x[1])

    # 输出结果
    for label, avg_dist, sentence_vector_list in cluster_avg_distance_list:
        print(f"cluster {label} (平均距离: {avg_dist:.2f}) :")
        for sentence, _ in sentence_vector_list[:10]:  # 输出不超过10个
            print(sentence.replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
