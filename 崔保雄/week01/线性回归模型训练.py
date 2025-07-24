# 基于Scikit-learn实现多元线性回归模型   
import pandas as pd 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#从数据集读取数据
filepath = "../造数据/data/test_003.csv"
df = pd.read_csv(filepath)
# print("df：\n", df)

#准备数据集
X = df[['X1', 'X2', 'X3']]
y = df[['Y1']]
# print("X：\n", X)
# print("y：\n", y)

#划分训练集和测试集，训练集80%，测试机20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X_train：\n", X_train)
# print("X_test：\n", X_test)
# print("y_train：\n", y_train)
# print("y_test：\n", y_test)

# 创建线性回归模型并训练它
model = LinearRegression()
model.fit(X_train, y_train)

# 评估模型性能、使用测试集进行预测
y_pred = model.predict(X_test)
 
# 计算并打印均方误差（MSE）和R^2分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 打印模型参数
a = model.coef_
b = model.intercept_
print("线性回归模型为： y = {:.4f} * x1 + {:.4f} * x2 + {:.4f}* x3 + {:.4f}."
      .format(a[0][0],a[0][1],a[0][2],b[0]))
