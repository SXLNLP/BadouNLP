import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据集
file_path = "D:\\codeLearning\\code\\badouweek7\\文本分类练习.csv"
data = pd.read_csv(file_path)

# 2. 划分数据集，90% 训练，10% 测试，并确保标签分布一致
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42, stratify=data['label'])

# 3. 保存训练集和测试集到新的 CSV 文件
train_data.to_csv("D:\\codeLearning\\code\\badouweek7\\训练集.csv", index=False, encoding='utf-8-sig')
test_data.to_csv("D:\\codeLearning\\code\\badouweek7\\测试集.csv", index=False, encoding='utf-8-sig')

print("训练集和测试集已成功保存！")