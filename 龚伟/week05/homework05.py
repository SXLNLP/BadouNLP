import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# 示例数据（替换为你的实际数据）
X = np.random.rand(100, 2)  # 100个二维样本点

# 使用KMeans聚类（假设分为3个簇）
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 计算每个簇的类内平均距离
cluster_distances = []
for i in range(len(centers)):
    # 获取当前簇的所有样本
    cluster_samples = X[labels == i]

    # 计算当前簇内每个样本到质心的距离
    distances = np.linalg.norm(cluster_samples - centers[i], axis=1)

    # 计算平均距离
    avg_distance = np.mean(distances)
    cluster_distances.append((i, avg_distance, len(cluster_samples)))

# 按平均距离从小到大排序
sorted_clusters = sorted(cluster_distances, key=lambda x: x[1])

# 输出排序结果
print("簇排序结果（按紧密程度升序）：")
print("簇索引 | 平均距离 | 样本数量")
print("-" * 30)
for cluster in sorted_clusters:
    print(f"  {cluster[0]}    | {cluster[1]:.4f}  |   {cluster[2]}")
