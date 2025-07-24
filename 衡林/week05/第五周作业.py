import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 加载词向量模型
def load_word2vec_model(model_path):
    print(f"正在加载词向量模型从 {model_path}...")
    model = Word2Vec.load(model_path)
    print("模型加载完成!")
    return model


# 加载句子数据
def load_sentences(file_path):
    print(f"正在从 {file_path} 加载句子...")
    sentences = set()  # 使用集合来避免重复句子

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    cut_words = jieba.cut(line)
                    sentence_with_spaces = " ".join(cut_words)
                    sentences.add(sentence_with_spaces)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到!")
        return set()

    print(f"成功加载 {len(sentences)} 条句子")
    return sentences


# 将句子转换为向量
def sentences_to_vectors(sentences, model):
    print("正在将句子转换为向量...")
    vectors = []
    vector_size = model.vector_size  # 获取向量的维度

    for sentence in sentences:
        words = sentence.split()  # 将句子拆分为单词列表
        sentence_vector = np.zeros(vector_size)  # 创建全零向量

        word_count = 0
        for word in words:
            try:
                sentence_vector += model.wv[word]  # 将每个词的向量相加
                word_count += 1
            except KeyError:  # 如果词不在词汇表中
                # 忽略该词(保持加0)
                pass

        # 计算平均向量(避免除以零)
        if word_count > 0:
            sentence_vector = sentence_vector / word_count

        vectors.append(sentence_vector)

    print("向量转换完成!")
    return np.array(vectors)


# 计算类内平均距离
def calculate_cluster_distance(vectors, indices):
    if len(indices) <= 1:
        return 0.0  # 只有一个点，距离为0

    total_distance = 0.0
    pair_count = 0

    # 获取该簇的所有向量
    cluster_vectors = vectors[indices]

    # 计算所有点对之间的距离
    for i in range(len(cluster_vectors)):
        for j in range(i + 1, len(cluster_vectors)):
            # 计算欧几里得距离
            distance = np.linalg.norm(cluster_vectors[i] - cluster_vectors[j])
            total_distance += distance
            pair_count += 1

    # 计算平均距离
    if pair_count > 0:
        return total_distance / pair_count
    else:
        return 0.0


def main():
    # 1. 加载模型和数据
    model_path = "model.w2v"  # 模型文件路径
    sentences_file = "titles.txt"  # 句子文件路径

    print("=== 开始聚类分析 ===")

    # 加载词向量模型
    word2vec_model = load_word2vec_model(model_path)

    # 加载并分词句子
    all_sentences = load_sentences(sentences_file)
    if not all_sentences:
        print("没有加载到句子，程序退出!")
        return

    # 2. 转换为向量
    sentence_vectors = sentences_to_vectors(all_sentences, word2vec_model)

    # 3. 计算聚类数量(使用平方根规则)
    num_clusters = int(math.sqrt(len(all_sentences)))
    print(f"\n计算出的聚类数量: {num_clusters}")

    # 4. 进行KMeans聚类
    print("正在进行KMeans聚类...")
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sentence_vectors)
    print("聚类完成!")

    # 5. 组织聚类结果
    # 创建一个字典来存储每个簇的句子
    cluster_to_sentences = defaultdict(list)
    # 创建一个字典来存储每个簇的索引
    cluster_to_indices = defaultdict(list)

    # 填充字典
    for index, (sentence, cluster_id) in enumerate(zip(all_sentences, kmeans.labels_)):
        cluster_to_sentences[cluster_id].append(sentence)
        cluster_to_indices[cluster_id].append(index)

    # 6. 计算并排序簇
    clusters_info = []

    for cluster_id in cluster_to_sentences.keys():
        # 计算该簇的平均距离
        indices = cluster_to_indices[cluster_id]
        avg_distance = calculate_cluster_distance(sentence_vectors, indices)

        # 获取该簇的句子数量
        num_sentences = len(cluster_to_sentences[cluster_id])

        # 存储信息
        clusters_info.append({
            'cluster_id': cluster_id,
            'avg_distance': avg_distance,
            'num_sentences': num_sentences,
            'sentences': cluster_to_sentences[cluster_id]
        })

    # 按平均距离排序
    clusters_info.sort(key=lambda x: x['avg_distance'])

    # 7. 打印结果
    print("\n=== 聚类结果 ===")
    print(f"总簇数: {len(clusters_info)}")
    print("按类内平均距离从低到高排序:")

    for cluster in clusters_info:
        print(f"\n簇 {cluster['cluster_id']}:")
        print(f"类内平均距离: {cluster['avg_distance']:.4f}")
        print(f"包含句子数: {cluster['num_sentences']}")
        print("示例句子(最多10个):")

        # 打印前10个句子(去掉分词空格)
        for sentence in cluster['sentences'][:10]:
            print(sentence.replace(" ", ""))

        print("-" * 40)


if __name__ == "__main__":
    main()
