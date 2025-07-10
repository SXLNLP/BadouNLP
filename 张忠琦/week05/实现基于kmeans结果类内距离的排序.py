
import math
import re # 这个模块在此代码中未使用，可以删除
import json # 这个模块在此代码中未使用，可以删除
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances # 导入计算距离的函数

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    # 使用列表而不是集合，因为我们需要保持句子和其向量的对应顺序
    sentences = [] 
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            if sentence: # 避免处理空行
                sentences.append(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        if not words: # 避免空句子导致ZeroDivisionError
            vector = np.zeros(model.vector_size)
        else:
            vector = np.zeros(model.vector_size)
            #所有词的向量相加求平均，作为句子向量
            for word in words:
                try:
                    vector += model.wv[word]
                except KeyError:
                    #部分词在训练中未出现，用全0向量代替
                    vector += np.zeros(model.vector_size)
            vector /= len(words) # 求平均
        vectors.append(vector)
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v") #加载词向量模型

    sentences_list = load_sentence("titles.txt")  #加载所有标题，现在返回的是一个list
    vectors = sentences_to_vectors(sentences_list, model)   #将所有标题向量化

    if len(vectors) == 0:
        print("没有可用的句子向量进行聚类。")
        return

    n_clusters = int(math.sqrt(len(sentences_list)))  #指定聚类数量
    if n_clusters == 0: # 避免只有一条句子导致n_clusters为0
        n_clusters = 1
    elif n_clusters > len(sentences_list): # 避免聚类数量大于样本数量
        n_clusters = len(sentences_list)
        
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)  
    kmeans.fit(vectors)          #进行聚类计算

    cluster_data_with_distances = defaultdict(list)

    for i, vec in enumerate(vectors):
        label = kmeans.labels_[i]
        original_sentence = sentences_list[i] # 获取原始句子文本
        centroid = kmeans.cluster_centers_[label] # 获取该句子所属簇的质心
        # euclidean_distances expects 2D arrays, so we pass [vec] and [centroid]
        distance = euclidean_distances([vec], [centroid])[0][0]
        cluster_data_with_distances[label].append((distance, original_sentence))
    print("\n--- 聚类结果按类内距离排序 (距离越小越典型) ---")
    for label, items in cluster_data_with_distances.items():
        sorted_items = sorted(items, key=lambda x: x[0]) # x[0] 是距离
        print(f"\n簇 {label} (质心: {kmeans.cluster_centers_[label]}) 共 {len(items)} 条句子:")
        print("该簇内句子按与质心距离升序排序:")
        
        for i, (dist, sentence_text) in enumerate(sorted_items):
            if i >= 10: # 只打印前10个句子，可以根据需要调整数量
                break
            print(f"  距离: {dist:.4f} - {sentence_text.replace(' ', '')}")
        print("-" * 30)

if __name__ == "__main__":
    main()
