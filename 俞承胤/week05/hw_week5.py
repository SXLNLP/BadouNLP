
import jieba
import numpy as np
import math

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vectors_label_dict = defaultdict(list)
    average_vectors_label_dict = defaultdict(list)
    # kmeans.cluster_centers_[0]
    for sentence, label, vector  in zip(sentences, kmeans.labels_,vectors):
        # print("cluster_centers %s :%s %f" % (label, sentence.replace(" ", "") ,np.linalg.norm(kmeans.cluster_centers_[label] - vector)))
        sentence_label_dict[label].append(sentence.replace(" ", ""))         #同标签的放到一起
        vectors_label_dict[label].append(np.linalg.norm(kmeans.cluster_centers_[label] - vector))

    print(vectors_label_dict)

    for label, vectors in vectors_label_dict.items():
        # print("cluster %s %f:" % (label,np.mean(vectors)))
        average_vectors_label_dict[label].append(np.mean(vectors))

    sorted_label_by_vectors = dict(sorted(average_vectors_label_dict.items(), key=lambda item: item[1]))
    print(sorted_label_by_vectors)

    top10_lable_list=list(sorted_label_by_vectors.keys())[:10]
    print(top10_lable_list)

    for label in top10_lable_list:
        print("cluster %s :" % label)
        print(sentence_label_dict[label])


    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    print("---------")



if __name__ == "__main__":
    main()
