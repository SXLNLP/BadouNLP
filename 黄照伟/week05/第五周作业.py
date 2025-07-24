import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from collections import defaultdict

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path,encoding='utf-8') as f:
        for line in f:
            sentence =line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量:",len(sentences))
    return sentences

#将文本向量化
def sentence_to_vector(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector/len(words))
    return np.array(vectors)

def main():
    """
    主函数：执行文本向量化聚类全流程
    Args:
        无
    Returns:
        无
    """
    # 加载预训练的Word2Vec模型用于文本向量化
    model = load_word2vec_model("model.w2v")

    # 从文件加载待处理的标题文本数据
    sentences = load_sentence("titles.txt")

    # 将文本列表转换为对应的向量表示
    vectors = sentence_to_vector(sentences, model)

    # 根据样本数量动态确定聚类数（平方根法则）
    n_clusters = int(math.sqrt(len(sentences)))
    print("制定聚类数量:", n_clusters)

    # 初始化KMeans聚类器并进行训练
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    # 构建句子标签映射字典（标签 -> 句子列表）
    sentence_label_dict = defaultdict(list)

    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_
    # 使用 欧式距离
    for sentence,vector, label in zip(sentences, vectors,kmeans.labels_):
        # todo: 计算kmeans.labels_ 的类内平均中心距离
        # todo: 按照平均距离进行排序, 输出句子

        center = cluster_centers[label]
        distance = np.linalg.norm(vector - center)  # 计算欧氏距

        sentence_label_dict[label].append((sentence,distance))
    for label , sentences in sentence_label_dict.items():
        print("cluster %s:" %label)
        sentences = sorted(sentences,key=lambda x:x[1])
        for i in range(min(10,len(sentences))):
            print(sentences[i][0].replace(" ", ""))
        print("-----------------")

    # #使用余弦距离
    # for sentence ,vector,label in zip(sentences,vectors,kmeans.labels_):
    #     center = cluster_centers[label]
    #     distance = cosine(vector,center)
    #     sentence_label_dict[label].append((sentence,distance))
    #
    # for label,sentences in sentence_label_dict.items():
    #     print("cluster %s:" %label)
    #     # 按照距离 从小到大进行排序
    #     sentences = sorted(sentences, key=lambda x: x[1])
    #     for i in range(min(10,len(sentences))):
    #         print(sentences[i][0].replace(" ", ""))


    # # 按标签分类输出聚类结果（每个类别最多显示10个样本）
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s:" % label)
    #     for i in range(min(10, len(sentences))):
    #         print(sentences[i].replace(" ", ""))
    #     print("-----------------")

if __name__ == "__main__":
    main()

