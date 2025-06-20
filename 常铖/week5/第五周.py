from sklearn.cluster import KMeans
import numpy as np

# 示例数据集
data = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
])

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# 计算每个点到其簇中心的距离
distances = kmeans.transform(data)
min_distances = distances.min(axis=1)

# 将数据和对应的最小距离组合在一起
data_with_distances = list(zip(data, min_distances))

# 按照距离从小到大排序
sorted_data = sorted(data_with_distances, key=lambda x: x[1])

# 输出排序后的结果
for point, distance in sorted_data:
    print(f"Point: {point}, Distance to cluster center: {distance}")
