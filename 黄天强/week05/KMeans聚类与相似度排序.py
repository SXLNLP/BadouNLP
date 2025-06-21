import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Dict, List, Tuple

"""
基于训练好的词向量模型对句子进行KMeans聚类，并根据聚类中心的密集程度进行排序输出。
"""


def load_word2vec_model(model_path: str) -> Word2Vec:
    """加载训练好的 Word2Vec 模型"""
    return Word2Vec.load(model_path)


def load_sentence(file_path: str):
    """读取文本文件中的句子，进行分词并去重"""
    sentences = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                tokenized = " ".join(jieba.cut(sentence))
                sentences.add(tokenized)
    print(f"获取句子数量：{len(sentences)}")
    return sentences


def sentences_to_vectors(sentences, model: Word2Vec):
    """将分好词的句子转换为句向量（平均词向量）"""
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vec = np.zeros(model.vector_size)
        valid_words = 0

        for word in words:  # 所有词的向量相加求平均，作为句子向量
            if word in model.wv:
                vec += model.wv[word]  # 累加词向量
                valid_words += 1

        if valid_words > 0:
            vec /= valid_words
        vectors.append(vec)

    return np.array(vectors)


def compute_cluster_cosine_densities(
        vectors: np.ndarray,
        kmeans: KMeans
) -> List[Tuple[int, np.floating]]:
    """
    计算每个聚类的平均类内余弦相似度
    """
    cluster_similarities: Dict[int, List[float]] = defaultdict(list)

    for idx, label in enumerate(kmeans.labels_):
        vector = vectors[idx].reshape(1, -1)
        centroid = kmeans.cluster_centers_[label].reshape(1, -1)
        similarity = cosine_similarity(vector, centroid)[0, 0]
        cluster_similarities[label].append(similarity)

    # 计算每个聚类的平均相似度
    average_similarities: Dict[int, np.floating] = {
        label: np.mean(similarities)
        for label, similarities in cluster_similarities.items()
    }

    # 按平均相似度降序排列
    sorted_densities = sorted(average_similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_densities


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 设置聚类数量为 √N
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters, random_state=42)  # 初始化KMeans
    kmeans.fit(vectors)  # 执行聚类

    sentence_label_dict = defaultdict(list)  # 构建聚类标签对应的句子集合
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    densities = compute_cluster_cosine_densities(
        vectors=vectors,
        kmeans=kmeans,
    )

    # 按聚类密集度顺序输出每类中的前10个句子（示例）
    for label, avg_sim in densities:
        print(f"Cluster {label} | Avg Similarity: {avg_sim:.4f}")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(f"  {i + 1:>2}. {sentences[i].replace(' ', '')}")
        print("---------")


if __name__ == "__main__":
    main()
