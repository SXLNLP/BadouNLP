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

def compute_distance(center_vect, sample_vects):
    avg_distances = np.sqrt(np.sum((center_vect - sample_vects) ** 2, axis=1))
    # avg_distances = np.linalg.norm(sample_vects - center_vect, axis=1)
    return np.mean(avg_distances)

def label_distance_order(distance_list):
    print('按平均距离“升序”排序 ============')
    return sorted(distance_list, key=lambda content: content['avg_distance'])

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    label_res_dict = defaultdict(
        lambda: {'sentences': [], 'vectors': [], 'center_vect': None}
    )
    order_label_list = []

    # kmeans.labels_ 表示传入的 sentences 转换为向量后，每一个向量所属的簇类标签的集合，其顺序与传入的 vectors 顺序一一对应，又因为 vectors 与 sentences 对应，所以三者顺序一一对应；
    for sentence, label, vector in zip(sentences, kmeans.labels_, vectors):  #取出句子和标签
        label_res_dict[label]['sentences'].append(sentence)         #同标签标题的放到一起
        label_res_dict[label]['vectors'].append(vector)         #同标签向量放到一起

    # print('kmeans.cluster_centers_:\n', kmeans.cluster_centers_)
    for index, vector in enumerate(kmeans.cluster_centers_):
        # 类的中心向量
        label_res_dict[index]['center_vect'] = vector

    # print(f'label_res_dict[0][center_vect]:\n{label_res_dict[0]['center_vect']}')
    for label, label_content in label_res_dict.items():
        vectors, center_vect = label_content['vectors'], label_content['center_vect']
        order_label_list.append({
            'label': label,
            # 'center_cect': center_vect,
            'avg_distance': compute_distance(np.array(center_vect), vectors),
        })

    order_label_list = label_distance_order(list(order_label_list))
    print(f'order_label_list 长度：{len(order_label_list)}')
    print(f'order_label_list:\n{order_label_list}')
if __name__ == "__main__":
    main()
