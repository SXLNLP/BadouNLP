import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
np.random.seed(42)


def calculate_intra_cluster_distance(X, labels, cluster_id):
    """计算单个簇的类内平均距离"""
    cluster_points = X[labels == cluster_id]
    if len(cluster_points) <= 1:
        return 0.0

    # 计算簇内所有点对之间的平均距离
    distances = []
    for i in range(len(cluster_points)):
        for j in range(i + 1, len(cluster_points)):
            distances.append(np.linalg.norm(cluster_points[i] - cluster_points[j]))

    return np.mean(distances) if distances else 0.0


def main():
    # 生成样本数据
    X, y = make_blobs(n_samples=300, centers=5, cluster_std=[1.0, 1.5, 0.5, 2.0, 1.2], random_state=42)

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)

    # 计算每个簇的类内距离
    cluster_distances = []
    for cluster_id in range(kmeans.n_clusters):
        distance = calculate_intra_cluster_distance(X, labels, cluster_id)
        cluster_distances.append((cluster_id, distance))

    # 按类内距离从大到小排序
    sorted_clusters = sorted(cluster_distances, key=lambda x: x[1], reverse=True)

    # 输出排序结果
    print("按类内距离排序的簇（从大到小）：")
    for cluster_id, distance in sorted_clusters:
        print(f"簇 {cluster_id}: 类内平均距离 = {distance:.4f}")

    # 可视化聚类结果
    plt.figure(figsize=(12, 10))

    # 绘制原始数据点，按聚类结果着色
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k', alpha=0.7)

    # 标记聚类中心
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, label='聚类中心')

    # 添加排序后的类内距离信息
    for i, (cluster_id, distance) in enumerate(sorted_clusters):
        plt.annotate(f'簇 {cluster_id}: {distance:.2f} (#{i + 1})',
                     (centers[cluster_id, 0], centers[cluster_id, 1]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title('K-means聚类结果及类内距离排序')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)


if __name__ == "__main__":
    main()
