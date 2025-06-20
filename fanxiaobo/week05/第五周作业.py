import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def sort_by_intra_distance(X, n_clusters=3):
    """
    基于KMeans聚类结果，按类内平均距离排序聚类中心

    参数:
    X: 输入数据 (n_samples, n_features)
    n_clusters: 聚类数量

    返回:
    sorted_centers: 按类内距离排序后的中心点
    sorted_labels: 重新排序后的标签
    intra_distances: 各类的平均类内距离
    """
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # 计算每个类的类内平均距离
    intra_distances = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            dist = pairwise_distances(cluster_points, [centers[i]]).mean()
            intra_distances.append(dist)
        else:
            intra_distances.append(0)

    # 按类内距离排序
    sorted_indices = np.argsort(intra_distances)[::-1]  # 从大到小排序
    sorted_centers = centers[sorted_indices]
    sorted_intra_distances = np.array(intra_distances)[sorted_indices]

    # 重新映射标签
    label_mapping = {old: new for new, old in enumerate(sorted_indices)}
    sorted_labels = np.array([label_mapping[l] for l in labels])

    return sorted_centers, sorted_labels, sorted_intra_distances


# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # 生成测试数据
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    # 执行排序
    centers, labels, distances = sort_by_intra_distance(X, n_clusters=4)

    print("排序后的中心点:\n", centers)
    print("各类平均类内距离:", distances)
