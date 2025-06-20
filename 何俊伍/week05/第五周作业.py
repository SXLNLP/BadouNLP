import numpy as np
import random
import sys
'''
Kmeans算法实现
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
    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum
    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)
    def __distance(self, p1, p2):
        #计算两点间距
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
def sort_clusters_by_intra_distance(result, centers):
    #按类内总距离从小到大排序聚类结果
    intra_dists = []
    for i in range(len(result)):
        center = centers[i]
        dist_sum = sum(np.linalg.norm(np.array(p) - center) for p in result[i])
        intra_dists.append(dist_sum)
    combined = list(zip(result, centers, intra_dists))
    combined.sort(key=lambda x: x[2])  # 类内距离升序
    result_sorted, centers_sorted, dists_sorted = zip(*combined)
    return list(result_sorted), np.array(centers_sorted), list(dists_sorted)
x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
result, centers, intra_distances = sort_clusters_by_intra_distance(result, centers)
for i, cluster in enumerate(result):
    print(f"Cluster {i}: size={len(cluster)}, intra-distance={intra_distances[i]:.4f}")
