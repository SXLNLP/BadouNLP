#该代码用来手动造一些数据集，用于后续线性回归模型的训练。   

import numpy as np
import pandas as pd

#定义3个参数。值随便写，后续基于生成的数据集，使用线性回归模型算法，看能不能找到比较接近这三个参数的模型参数。
w1 = 0.085
w2 = 0.015
w3 = 0.05

#生成的样本数量
data_size = 6000

#定义三个x，随机赋值
x_old = np.array( np.random.randint(1000 , 10000, size = (data_size, 3)) )
# print("x_old的行数：\n", x1.shape[0])
# print("x_old：\n",x_old)

#定义一个y，根据多元一次函数生成。计算y值的时候，故意加了一个随机整数模拟日常生活当中数据的复杂情况。
new_column =np.array(  [ np.round(w1 * i[0] + w2 * i[1] + w3 * i[2] + np.random.randint(-100 , 100), 4)  for i in x_old]  )
print("new_column：\n", new_column)
# print("new_column的行数：\n", new_column.shape[0])

#将三位数组扩编，变成4维数组  reshape的第一个参数3000代表行数
data_new = np.hstack( (x_old, new_column.reshape(data_size, 1)) )
# print("data_new：\n", data_new)

filepath = "data/test_003.csv"
#给数据集字段加个标题
data_title = np.array(["X1", "X2", "X3", "Y1"])
data_source = np.vstack((data_title, data_new))
print("dt_source：\n", data_source)

df = pd.DataFrame(data_source)

#开始保存文件
#index、header 用来设置是否显示行索引列索引
df.to_csv(filepath, index = False, header = False)
