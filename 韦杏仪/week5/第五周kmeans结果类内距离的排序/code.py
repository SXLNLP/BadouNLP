import numpy as np
import random
import sys


class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        cluster_distances = []  # 新增：记录每个簇的距离
        for i in range(self.cluster_num):
            result.append([])
            cluster_distances.append([])  # 初始化距离记录

        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
            cluster_distances[index].append(distance_min)  #记录距离
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            #计算平均距离并排序
            avg_distances = [np.mean(dist) if dist else 0 for dist in cluster_distances]
            sorted_indices = np.argsort(avg_distances)

            return result, self.points, sum, sorted_indices

        self.points = np.array(new_center)
        return self.cluster()


    def __sumdis(self, result):
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
        return sum

    def __center(self, list):
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)



x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 5)
result, centers, total_distance, sorted_indices = kmeans.cluster()
print("按类内距离排序的簇索引(从紧凑到分散):", sorted_indices)
print("各类点数:", [len(cluster) for cluster in result])
print("总距离和:", total_distance)
