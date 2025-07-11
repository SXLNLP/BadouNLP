import random

"""
    本示例代码，用于切割样本数据，将原样本数据按照指定比例，随机拆分为训练集、测试集，并分开保存
"""

#读取文件
lines = []

with open("data/文本分类练习.csv", encoding="utf-8") as file:
    lines = file.readlines()

line_title= lines[0:1]
lines = lines[1:]
#随机打乱顺序
random.shuffle(lines)

# 拆分数据为训练集和测试集，例如80%训练，20%测试
train_size = int(len(lines) * 0.8)
train_data = lines[:train_size]
test_data = lines[train_size:]

#保存文件-训练集
with open("data/文本分类练习_train.csv", "w", encoding="utf-8") as file:
    #最前面再加上标题行
    file.writelines(line_title)

    file.writelines(train_data)

#保存文件-测试集
with open("data/文本分类练习_test.csv", "w", encoding="utf-8") as file:
    # 最前面再加上标题行
    file.writelines(line_title)

    file.writelines(test_data)
