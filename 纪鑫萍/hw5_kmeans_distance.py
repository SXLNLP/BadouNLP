      
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
    sentences = set()
    # sentences = list()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
            # sentences.append(" ".join(jieba.cut(sentence)))
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
    model = load_word2vec_model(r"/Users/juju/nlp20/class5/model.w2v")  #加载词向量模型
    sentences = load_sentence("/Users/juju/nlp20/class5/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)  #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)  #进行聚类计算

    sentence_label_dict = defaultdict(list)
    cos_label_dict = defaultdict(list)
    for index, label in enumerate(kmeans.labels_):  # ❗
        center_vector = kmeans.cluster_centers_[label]  # 获取聚类中心的向量
        sentence_vector = vectors[index]  # ❗
        cos = np.dot(center_vector, sentence_vector) / (np.sqrt(np.sum(np.square(sentence_vector))) * np.sqrt(np.sum(np.square(center_vector))))
        cos_label_dict[label].append(cos)

    # 转化为平均距离
    for label, cos_list in cos_label_dict.items():
        cos_label_dict[label] = np.mean(cos_list)

    # 根据平均距离正序排序
    sorted_cos_items = sorted(cos_label_dict.items(), key=lambda x: x[1])

    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)  #同标签的放到一起

    for label, distance in sorted_cos_items:
        label_sentences = sentence_label_dict[label]
        print("cluster %s - 【平均距离：%s】:" % (label, distance))
        for i in range(min(10, len(label_sentences))):  #随便打印几个，太多了看不过来
            print(label_sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
