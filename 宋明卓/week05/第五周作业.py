import numpy as np
import jieba
import math
import json
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def lode_model(path):
    model = Word2Vec.load(path)
    return model


# 加载所有标题
def lode_sentence(path):
    sentences = []
    seen = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            seg_sentence = ' '.join(jieba.cut(word))
            if seg_sentence not in seen:
                sentences.append(seg_sentence)
                seen.add(seg_sentence)
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本转换成向量
def sentences_to_vector(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:  # 如果该文本不存在
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = lode_model(r'model.w2v')  # 加载词向量模型
    sentences = lode_sentence('titles.txt')  # 加载所有标题
    vectors = sentences_to_vector(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))
    # 实例化
    kmeans = KMeans(n_clusters)
    # 模型训练
    kmeans.fit(vectors)

    distances_vector_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  # 取出句子、标签、标题向量
        # 计算类内距离
        center = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(vector - center)
        distances_vector_dict[label].append((distance, sentence))

    # 类内排序
    for label in distances_vector_dict:
        distances_vector_dict[label].sort(key=lambda x: x[0])

    # 类间排序
    cluster_quality = []
    for label, entries in distances_vector_dict.items():
        avg_dist = sum(a[0] for a in entries) / len(entries)
        cluster_quality.append((avg_dist, label, entries))
    cluster_quality.sort()

    for avg_dist, label, entries in sorted(cluster_quality, key=lambda x: x[1]):
        print(f"\nCluster {label} 平均距离：{avg_dist:.4f}")
        for dist, sent in entries[:10]:  # 显示前10个最近的
            print(sent.replace(" ", ""), f"({dist:.2f})")
        print('==' * 20)


if __name__ == '__main__':
    main()
