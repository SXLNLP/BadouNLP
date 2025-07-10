import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277

在原本代码的基础上添加了计算簇内平均距离的功能
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


def sorted_cluster_result(cluster_num,  result, centers):
    sorted_dis = []
    for i in range(cluster_num):
        center_of_mass = centers[i]
        item_num = 0
        dis_sum = 0
        for item in result[i]:
            dis_sum += distance(center_of_mass, item)
            item_num += 1
        mean_dist = dis_sum / item_num
        sorted_dis.append(mean_dist)
    return sorted(sorted_dis)


def distance(p1, p2):
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


x = np.random.rand(100, 8)
cluster_num = 10
kmeans = KMeansClusterer(x, cluster_num)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
sorted_cluster_dist = sorted_cluster_result(cluster_num, result, centers)
for i, mean_dist in enumerate(sorted_cluster_dist, start=1):
    print(print(f"第{i}簇分类的平均距离: {mean_dist:.4f}"))
