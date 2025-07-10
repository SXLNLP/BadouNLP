import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
raw_data = pd.read_csv("文本分类练习.csv")

# 检查数据
print("原始数据样例：")
print(raw_data.head())

# 划分训练集和验证集 (80%训练，20%验证)
train_df, valid_df = train_test_split(
    raw_data,
    test_size=0.2,
    random_state=42,
    stratify=raw_data['label']  # 保持标签分布一致
)

# 保存文件
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("validate_data.csv", index=False)

print(f"\n训练集样本数: {len(train_df)}")
print(f"验证集样本数: {len(valid_df)}")
