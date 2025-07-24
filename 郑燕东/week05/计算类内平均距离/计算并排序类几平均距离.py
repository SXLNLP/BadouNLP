import jieba
import math
import re
import json
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path,encoding="utf8") as f:
        for line in f :
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子的数量:",len(sentences))
    return sentences

def sentences_to_vectors(sentences,model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except:
                vector += np.zeros(model.vector_size)
            vectors.append(vector/len(words))
    return np.array(vectors)

def calculate_intra_cluster_distances(vectors,labels,centers):
    cluster_distances = defaultdict(list)
    for vec,label in zip(vectors,labels):
        dist = np.linalg.norm(vec-centers[label])
        cluster_distances[label].append(dist)
    avg_distances = {
        label:np.mean(distances)
        for label,distances in cluster_distances.items()
    }
    return avg_distances

def main():
   model = load_word2vec_model(r"E:\09-python\04-八斗课件\week5词向量及文本向量\model.w2v")
   sentences = load_sentence("titles.txt")
   vectors = sentences_to_vectors(sentences,model)
   n_clusters = int(math.sqrt(len(sentences)))
   print("指定聚类数量：",len(sentences))
   kmeans = KMeans(n_clusters)
   kmeans.fit(vectors)
   sentence_label_dict = defaultdict(list)
   for sentence,label in zip(sentences,kmeans.labels_):
       sentence_label_dict[label].append(sentence)
   for label,sentences in sentence_label_dict.items():
       print("cluster:",len(sentences))
       for i in range(min(10,len(sentences))):
           print(sentences[i].replace(" ",""))
   #新增距离计算
   centers = kmeans.cluster_centers_
   avg_distances = calculate_intra_cluster_distances(vectors,kmeans.labels_,centers)
   #按平均距离排序
   sorted_clusters = sorted(avg_distances.items(),key=lambda x:x[1])
   print("\n类内平均距离排序结果：")
   for label,avg_dist in sorted_clusters:
       print(f"簇{label}: 平均距离 = {avg_dist:.4f}")
       print(f"代表样本：{sentence_label_dict[label][0][:50]}...")
       print("------")
if __name__=="__main__":
    main()
