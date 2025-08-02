import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv("文本分类练习.csv")

# 检查标签分布
print("标签分布统计:")
print(df['label'].value_counts(normalize=True))

# 分割数据集（80%训练集，20%验证集）
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,  # 随机种子确保可复现
    stratify=df['label']  # 保持类别比例
)

# 保存分割后的数据集
train_df.to_csv("train_dataset.txt", index=False)
val_df.to_csv("val_dataset.txt", index=False)

print("\n数据集分割完成:")
print(f"训练集大小: {len(train_df)} 条")
print(f"验证集大小: {len(val_df)} 条")
print(f"分割比例: {len(train_df)/len(df):.1%} 训练 / {len(val_df)/len(df):.1%} 验证")