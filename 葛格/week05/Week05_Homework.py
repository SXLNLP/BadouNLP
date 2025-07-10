"""
kmeans计算类内平均距离，只保留前70%的类
循环5次
"""

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_word2vec_model(path):
    return Word2Vec.load(path)

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    return list(sentences)

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vector /= len(words) if len(words) > 0 else 1
        vectors.append(vector)
    return np.array(vectors)

def compute_intra_cluster_distances(vectors, labels, centers):
    cluster_dists = defaultdict(list)
    for i, label in enumerate(labels):
        dist = np.linalg.norm(vectors[i] - centers[label])
        cluster_dists[label].append(dist)

    cluster_avg_dist = {
        label: np.mean(dists) for label, dists in cluster_dists.items()
    }
    return cluster_avg_dist

def main():
    model = load_word2vec_model("model.w2v")
    all_sentences = load_sentence("titles.txt")
    all_vectors = sentences_to_vectors(all_sentences, model)

    max_rounds = 5
    retain_ratio = 0.7
    current_sentences = all_sentences
    current_vectors = all_vectors

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== 第 {round_num} 轮聚类 ===")

        if len(current_sentences) < 10:
            print("样本过少，终止聚类。")
            break

        n_clusters = max(2, int(math.sqrt(len(current_sentences))))
        n_clusters = min(n_clusters, len(current_sentences) // 2)
        print("当前聚类数量：", n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(current_vectors)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        avg_dists = compute_intra_cluster_distances(current_vectors, labels, centers)

        sentence_label_dict = defaultdict(list)
        for i, (sentence, label) in enumerate(zip(current_sentences, labels)):
            sentence_label_dict[label].append((sentence, current_vectors[i]))

        # 剔除只有1条样本的簇
        sentence_label_dict = {
            label: sents for label, sents in sentence_label_dict.items()
            if len(sents) > 1
        }

        # 更新簇数
        effective_clusters = list(sentence_label_dict.keys())
        if len(effective_clusters) < 2:
            print("有效聚类簇过少，停止。")
            break

        # 计算并排序平均距离
        sorted_clusters = sorted(
            [(label, avg_dists[label]) for label in effective_clusters],
            key=lambda x: x[1]
        )

        retain_count = max(2, int(len(sorted_clusters) * retain_ratio))
        retained_labels = set(label for label, _ in sorted_clusters[:retain_count])
        print("保留簇数量：", retain_count)

        # 过滤保留样本
        filtered_sentences = []
        filtered_vectors = []
        retained_clusters = {}

        for label in retained_labels:
            sents = sentence_label_dict.get(label, [])
            for sentence, vec in sents:
                filtered_sentences.append(sentence)
                filtered_vectors.append(vec)
            retained_clusters[label] = [s for s, _ in sents]

        # 打印前5个非空聚类
        top_k_labels = [label for label, _ in sorted_clusters[:5]]
        for label in top_k_labels:
            sents = retained_clusters.get(label, [])
            if not sents:
                continue
            print(f"\nCluster {label} (类内距离: {avg_dists[label]:.4f}):")
            for s in sents[:5]:  # 最多打印5条
                print(s.replace(" ", ""))
            print("---------")

        # 判断是否收敛
        if len(filtered_sentences) == len(current_sentences):
            print("聚类结果收敛，提前终止。")
            break

        current_sentences = filtered_sentences
        current_vectors = np.array(filtered_vectors)

if __name__ == "__main__":
    main()
