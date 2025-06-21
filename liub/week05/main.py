# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/20
# @Author      : liuboyuan
# @Description :

from text_reader import TextReader
from text_cleaner import TextCleaner
from text_tokenizer import TextTokenizer
from text_vectorizer import TextVectorizer
from kmeans import KMeans
from visualizer import Visualizer
import numpy as np

def main():
    # 1. 读取文本
    reader = TextReader('data.txt')
    text = reader.read()
    if text is None:
        print("读取文件失败")
        return
        
    # 2. 将文本按段落分割（使用两个换行符分割）
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    print(f"总共找到 {len(paragraphs)} 个段落")
    
    # 3. 对每个段落进行处理
    cleaner = TextCleaner()
    tokenizer = TextTokenizer()
    
    # 存储处理后的段落和对应的原始段落
    cleaned_paragraphs = []
    tokenized_paragraphs = []
    valid_original_paragraphs = []
    
    for p in paragraphs:
        # 清洗文本
        cleaned_text = cleaner.clean(p)
        if not cleaned_text:  # 如果清洗后为空，跳过
            continue
            
        # 分词
        tokens = tokenizer.tokenize(cleaned_text)
        if len(tokens) < 3:  # 如果分词结果少于3个词，认为是无效段落
            continue
            
        cleaned_paragraphs.append(cleaned_text)
        tokenized_paragraphs.append(tokens)
        valid_original_paragraphs.append(p)
    
    print(f"清洗和分词后剩余 {len(cleaned_paragraphs)} 个有效段落")
    
    # 4. TF-IDF向量化
    vectorizer = TextVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_paragraphs)
    
    if len(tfidf_matrix) == 0:
        print("向量化失败")
        return
    
    # 5. K-Means聚类
    n_clusters = min(5, len(tfidf_matrix))  # 设置聚类数量
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(tfidf_matrix)
    
    # 6. 可视化
    visualizer = Visualizer()
    visualizer.visualize(tfidf_matrix, kmeans.labels)
    
    # 7. 输出聚类结果
    feature_names = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print(f"\n\n========== 簇 {i + 1} ==========")
        # 获取该簇的所有文档
        cluster_indices = [j for j, label in enumerate(kmeans.labels) if label == i]
        cluster_docs = tfidf_matrix[kmeans.labels == i]
        
        if len(cluster_docs) > 0:
            # 计算该簇的中心点
            centroid = np.mean(cluster_docs, axis=0)
            # 获取最重要的特征词（权重最大的词）
            top_features_idx = np.argsort(centroid)[-10:][::-1]
            print(f"\n主要特征词:")
            for idx in top_features_idx:
                print(f"{feature_names[idx]}", end=' ')
            
            print(f"\n\n该簇包含 {len(cluster_docs)} 个段落")
            print("\n示例段落（前2个）:")
            for idx in cluster_indices[:2]:
                print(f"\n原始段落:")
                print(valid_original_paragraphs[idx])
                print(f"\n清洗后的文本:")
                print(cleaned_paragraphs[idx])
                print("\n分词结果:")
                print(' '.join(tokenized_paragraphs[idx]))
                print("\n" + "="*50)

if __name__ == '__main__':
    main()
