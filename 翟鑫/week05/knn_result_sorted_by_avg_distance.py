import math
import random
import sys

import numpy
import numpy as np

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
        classified_points, centers, sum_distances = self.__cluster()

        # 计算平均的内类距离
        avg_distances = self.__avg_distance_in_class(classified_points, centers.tolist())
        # 带上类名称
        idx_avg_distances = zip(avg_distances, range(len(avg_distances)))
        # 根据平均distance 排序(decrease)
        sorted_avg_idx_distance = sorted(idx_avg_distances, key=lambda x: x[0], reverse=True)

        # 取出 idx
        sorted_idx = list(map(lambda x: x[1], sorted_avg_idx_distance))
        # 重排序分组的点
        classified_points = [classified_points[idx] for idx in sorted_idx]
        # 重排序分组的中心
        centers = [centers[idx] for idx in sorted_idx]
        
        return classified_points, centers, sum_distances

    def __avg_distance_in_class(self, points: list[list[list]], centers: list):

        group_avg_distances = []
        for group_id, group_points in enumerate(points):
            center = centers[group_id]

            sum = 0
            for point in group_points:
                dim_sum = 0
                for dim_idx, dim in enumerate(point):
                    dim_sum += pow(point[dim_idx] - center[dim_idx], 2)
                sum += math.sqrt(dim_sum)

            group_avg_distances.append(round(sum / len(group_points), 5))

        return group_avg_distances

    def __cluster(self):
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
        return self.__cluster()

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


x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
