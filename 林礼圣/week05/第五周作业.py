#week5作业
#实现基于kmeans结果类内距离的排序

import numpy as np
from sklearn.cluster import KMeans

# 示例数据 (替换为实际数据)
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# 训练KMeans模型 (n_clusters根据需求调整)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 计算每个样本到所属质心的距离
distances = np.sqrt(((X - centers[labels])**2).sum(axis=1))

# 存储排序结果的字典
sorted_clusters = {}
sorted_indices = {}  # 可选：存储原始索引

# 对每个簇进行内部排序
for cluster_idx in range(kmeans.n_clusters):
    # 获取当前簇的样本索引
    cluster_indices = np.where(labels == cluster_idx)[0]
    
    # 提取当前簇的距离并排序
    cluster_distances = distances[cluster_indices]
    sorted_order = np.argsort(cluster_distances)  # 升序排序索引
    
    # 按距离排序样本和索引
    sorted_clusters[cluster_idx] = X[cluster_indices][sorted_order]
    sorted_indices[cluster_idx] = cluster_indices[sorted_order]  # 原始索引

# 输出排序结果
for cluster_idx, samples in sorted_clusters.items():
    print(f"Cluster {cluster_idx} (按距离升序):")
    print(samples)
    print(f"样本索引: {sorted_indices[cluster_idx]}\n")
