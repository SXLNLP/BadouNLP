import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        K-Means聚类算法实现

        参数:
        n_clusters : 聚类数量，默认为3
        max_iter : 最大迭代次数，默认为300
        tol : 收敛阈值，默认为1e-4
        random_state : 随机种子，默认为None
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.cluster_distances = None

    def _initialize_centroids(self, X):
        """随机初始化质心"""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        """将点分配到最近的质心"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(centroids):
            # 计算每个点到质心的欧氏距离
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

        # 分配标签：每个点距离最近的质心索引
        labels = np.argmin(distances, axis=1)

        # 计算总距离（惯性）
        inertia = np.sum(np.min(distances, axis=1) ** 2)

        return labels, distances, inertia

    def _update_centroids(self, X, labels):
        """更新质心位置"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            # 计算每个簇的均值作为新质心
            cluster_points = X[labels == i]
            if cluster_points.shape[0] > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # 如果簇为空，重新随机初始化
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
        return new_centroids

    def fit(self, X):
        """训练K-Means模型"""
        # 初始化质心
        self.centroids = self._initialize_centroids(X)
        prev_inertia = None

        for iter in range(self.max_iter):
            # 分配点到最近的簇
            self.labels, distances, inertia = self._assign_clusters(X, self.centroids)
            self.inertia_ = inertia

            # 检查收敛
            if prev_inertia is not None and abs(prev_inertia - inertia) < self.tol:
                break

            # 更新质心
            new_centroids = self._update_centroids(X, self.labels)

            # 检查质心变化
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break

            self.centroids = new_centroids
            prev_inertia = inertia

        # 计算每个簇的类内平均距离
        self._compute_cluster_distances(X)

        return self

    def _compute_cluster_distances(self, X):
        """计算每个簇的类内平均距离"""
        self.cluster_distances = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if cluster_points.shape[0] == 0:
                self.cluster_distances.append(0)
                continue

            # 计算点到质心的距离
            distances = np.sqrt(np.sum((cluster_points - self.centroids[i]) ** 2, axis=1))
            self.cluster_distances.append(np.mean(distances))

    def sort_clusters_by_distance(self, ascending=True):
        """
        根据类内平均距离对簇进行排序

        参数:
        ascending : 是否升序排序（从最小平均距离开始），默认为True

        返回:
        sorted_clusters : 排序后的簇索引列表
        sorted_distances : 排序后的平均距离列表
        """
        if self.cluster_distances is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 获取排序索引
        sorted_indices = np.argsort(self.cluster_distances)
        if not ascending:
            sorted_indices = sorted_indices[::-1]

        # 返回排序结果
        sorted_clusters = sorted_indices.tolist()
        sorted_distances = [self.cluster_distances[i] for i in sorted_indices]

        return sorted_clusters, sorted_distances


# 生成示例数据的函数
def generate_data(n_samples=300, centers=4, cluster_std=1.0, random_state=None):
    """生成模拟聚类数据"""
    np.random.seed(random_state)
    # 随机生成质心位置
    centroids = np.random.uniform(-10, 10, size=(centers, 2))

    # 计算每个簇的样本数量
    cluster_sizes = np.random.multinomial(n_samples, np.ones(centers) / centers)

    # 生成每个簇的数据
    X = []
    labels = []
    for i, size in enumerate(cluster_sizes):
        # 围绕质心生成正态分布的点
        cluster_data = np.random.randn(size, 2) * cluster_std + centroids[i]
        X.append(cluster_data)
        labels.extend([i] * size)

    return np.vstack(X), np.array(labels)


# 可视化函数 - 修复后端问题
def plot_clusters(X, model, sorted_clusters, sorted_distances):
    """可视化聚类结果和排序"""
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 颜色映射
    colors = plt.cm.tab10.colors

    # 原始聚类结果
    for i in range(model.n_clusters):
        cluster_points = X[model.labels == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i % len(colors)],
                    alpha=0.6, label=f'Cluster {i}')

    # 绘制质心
    ax1.scatter(model.centroids[:, 0], model.centroids[:, 1],
                s=200, marker='X', c='black', label='Centroids')

    ax1.set_title('Original Clusters')
    ax1.legend()
    ax1.grid(True)

    # 按类内距离排序后的结果
    for rank, cluster_idx in enumerate(sorted_clusters):
        cluster_points = X[model.labels == cluster_idx]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[rank % len(colors)],
                    alpha=0.6,
                    label=f'Cluster {cluster_idx} (dist: {sorted_distances[rank]:.2f})')

    # 绘制质心
    ax2.scatter(model.centroids[:, 0], model.centroids[:, 1],
                s=200, marker='X', c='black', label='Centroids')

    ax2.set_title('Clusters Sorted by Intra-cluster Distance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 保存图像而不是直接显示
    plt.savefig('kmeans_clusters_sorted.png', dpi=150)
    print("聚类结果已保存为 'kmeans_clusters_sorted.png'")
    plt.close(fig)  # 关闭图形释放内存


# 主函数
def main():
    # 生成示例数据
    X, true_labels = generate_data(n_samples=300, centers=4, cluster_std=1.2, random_state=42)

    # 创建KMeans实例
    kmeans = KMeans(n_clusters=4, max_iter=300, tol=1e-4, random_state=42)

    # 训练模型
    start_time = time.time()
    kmeans.fit(X)
    training_time = time.time() - start_time

    # 根据类内平均距离排序
    sorted_clusters, sorted_distances = kmeans.sort_clusters_by_distance(ascending=True)

    # 打印结果
    print(f"训练完成，耗时: {training_time:.4f}秒")
    print(f"总惯性(SSE): {kmeans.inertia_:.4f}")
    print("\n簇按类内平均距离排序（从最紧凑开始）:")
    for rank, (cluster_idx, distance) in enumerate(zip(sorted_clusters, sorted_distances)):
        cluster_size = np.sum(kmeans.labels == cluster_idx)
        print(f"#{rank + 1} 簇 {cluster_idx}: 平均距离 = {distance:.4f}, 点数 = {cluster_size}")

    # 可视化结果
    plot_clusters(X, kmeans, sorted_clusters, sorted_distances)

    # 性能测试
    print("\n性能测试:")
    sizes = [100, 1000, 5000, 10000]
    for size in sizes:
        test_data, _ = generate_data(n_samples=size, centers=5, cluster_std=1.5)
        start_time = time.time()
        test_kmeans = KMeans(n_clusters=5, max_iter=100).fit(test_data)
        elapsed = time.time() - start_time
        print(f"数据集大小: {size}, 聚类时间: {elapsed:.4f}秒")


if __name__ == "__main__":
    main()
