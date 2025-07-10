import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        """初始化KMeans类
        
        Args:
            n_clusters (int): 聚类数量
            max_iters (int): 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        """训练K-Means模型
        
        Args:
            X (numpy.ndarray): 输入数据矩阵，shape为(n_samples, n_features)
        """
        if len(X) < self.n_clusters:
            raise ValueError("样本数量小于聚类数量")
            
        # 随机初始化聚类中心
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # 计算每个样本到聚类中心的距离
            distances = np.array([
                np.sqrt(np.sum((X - centroid) ** 2, axis=1))
                for centroid in self.centroids
            ])
            
            # 分配样本到最近的聚类中心
            new_labels = np.argmin(distances, axis=0)
            
            # 如果分类结果没有变化，则停止迭代
            if self.labels is not None and np.all(new_labels == self.labels):
                break
                
            self.labels = new_labels
            
            # 更新聚类中心
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
                    
    def predict(self, X):
        """预测新数据的聚类标签
        
        Args:
            X (numpy.ndarray): 输入数据矩阵，shape为(n_samples, n_features)
            
        Returns:
            numpy.ndarray: 聚类标签
        """
        distances = np.array([
            np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0) 