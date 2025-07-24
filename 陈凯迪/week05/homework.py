import numpy as np
import random
import sys

'''
Kmeans算法实现（添加类内距离排序功能）
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''


class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self, result):
        # 计算总距离和
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        # 计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)


# 计算单个簇的类内距离（所有点到中心的距离和）
def cluster_distance(cluster_points, center):
    return sum(np.linalg.norm(np.array(p) - np.array(center)) for p in cluster_points)


# 主程序
if __name__ == '__main__':
    x = np.random.rand(100, 8)
    kmeans = KMeansClusterer(x, 10)
    result, centers, total_distance = kmeans.cluster()

    # 计算每个簇的类内距离
    cluster_distances = []
    for i in range(len(result)):
        dist = cluster_distance(result[i], centers[i])
        cluster_distances.append(dist)

    # 根据类内距离排序（从小到大）
    sorted_indices = np.argsort(cluster_distances)
    sorted_result = [result[i] for i in sorted_indices]
    sorted_centers = [centers[i] for i in sorted_indices]
    sorted_distances = [cluster_distances[i] for i in sorted_indices]

    # 打印排序后的结果
    print("\n聚类结果按类内距离排序:")
    print(f"{'簇索引':<8} | {'样本数量':<8} | {'类内距离':<10} | {'中心点坐标'}")
    print("-" * 60)
    for i, (cluster, center, dist) in enumerate(zip(sorted_result, sorted_centers, sorted_distances)):
        print(f"{i:<8} | {len(cluster):<8} | {dist:<10.4f} | {np.round(center, 4)}")

    print("\n排序后的总距离和:", total_distance)
