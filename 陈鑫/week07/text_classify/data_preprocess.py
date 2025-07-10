import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取数据
data = pd.read_csv("./data/文本分类练习.csv", header=0)

# 2. 检查数据分布
print("原始数据分布:")
print(data["label"].value_counts(normalize=True))

# 3. 分层拆分数据集（保持标签比例）
train_data, test_data = train_test_split(
    data,
    test_size=0.8,                # 测试集占80%
    stratify=data["label"],       # 按标签分层抽样
    random_state=42               # 随机种子保证可复现
)

# 4. 验证拆分结果
print("\n训练集分布 (20%):")
print(train_data["label"].value_counts(normalize=True))
print("\n测试集分布 (80%):")
print(test_data["label"].value_counts(normalize=True))

# 5. 保存数据集
train_data.to_csv("./data/train_set.csv", index=False, header=False)
test_data.to_csv("./data/test_set.csv", index=False, header=False)

print("\n数据集已保存为 train_set.csv 和 test_set.csv")
