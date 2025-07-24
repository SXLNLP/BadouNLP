import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from collections import defaultdict
import math

# 输入模型文件路径
# 加载训练好的模型
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

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算类内平均欧式距离
def calculate_euclidean_distance(cluster_vectors):
    if len(cluster_vectors) <= 1:
        return 0
    distances = euclidean_distances(cluster_vectors)
    return np.sum(distances) / (len(cluster_vectors) * (len(cluster_vectors) - 1))

# 计算类内平均余弦距离
def calculate_cosine_distance(cluster_vectors):
    if len(cluster_vectors) <= 1:
        return 0
    distances = cosine_distances(cluster_vectors)
    return np.sum(distances) / (len(cluster_vectors) * (len(cluster_vectors) - 1))

def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 按聚类结果分组
    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
        vector_label_dict[label].append(vector)

    # 计算每个聚类的类内平均距离
    cluster_metrics = []
    for label in sentence_label_dict:
        cluster_vectors = np.array(vector_label_dict[label])
        euclidean_dist = calculate_euclidean_distance(cluster_vectors)
        cosine_dist = calculate_cosine_distance(cluster_vectors)
        cluster_metrics.append({
            'label': label,
            'euclidean_distance': euclidean_dist,
            'cosine_distance': cosine_dist,
            'size': len(cluster_vectors)
        })

    # 按距离排序
    euclidean_sorted_clusters = sorted(cluster_metrics, key=lambda x: x['euclidean_distance'], reverse=True)
    cosine_sorted_clusters = sorted(cluster_metrics, key=lambda x: x['cosine_distance'], reverse=True)

    # 输出距离统计信息
    print("\n按欧式距离排序的聚类：")
    for cluster in euclidean_sorted_clusters:
        print(f"聚类 {cluster['label']}: 大小={cluster['size']}, 欧式距离={cluster['euclidean_distance']:.4f}, 余弦距离={cluster['cosine_distance']:.4f}")

    print("\n按余弦距离排序的聚类：")
    for cluster in cosine_sorted_clusters:
        print(f"聚类 {cluster['label']}: 大小={cluster['size']}, 欧式距离={cluster['euclidean_distance']:.4f}, 余弦距离={cluster['cosine_distance']:.4f}")

    # 确定要舍弃的聚类（这里选择舍弃欧式距离最长的20%）
    discard_ratio = 0.2
    discard_count = max(1, int(len(cluster_metrics) * discard_ratio))
    discard_labels = [cluster['label'] for cluster in euclidean_sorted_clusters[:discard_count]]

    print(f"\n将舍弃以下聚类: {discard_labels}")

    # 输出保留的聚类
    print("\n保留的聚类:")
    for label in sentence_label_dict:
        if label not in discard_labels:
            print("cluster %s (大小=%d):" % (label, len(sentence_label_dict[label])))
            for i in range(min(10, len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
                print(sentence_label_dict[label][i].replace(" ", ""))
            print("---------")

if __name__ == "__main__":
    main()
