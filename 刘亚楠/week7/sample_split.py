import pandas as pd
from config import Config
# 读取数据（自动跳过首行标题）
df = pd.read_csv('../h_data/classification.csv', header=0)

# 随机打乱数据（设置random_state保证可复现性）
shuffled_df = df.sample(frac=1, random_state=Config["seed"]).reset_index(drop=True)

# 分割数据集（前80%训练，后20%测试）
split_point = int(len(shuffled_df) * 0.8)
train_df = shuffled_df.iloc[:split_point]
test_df = shuffled_df.iloc[split_point:]

# 进行描述性统计 查看label分布
# 统计训练集标签分布
train_label_stats = {
    "总样本数": len(train_df),
    "唯一值数量": train_df['label'].nunique(),
    "类别分布": train_df['label'].value_counts().to_dict(),
    "类别比例": train_df['label'].value_counts(normalize=True).to_dict()
}
# 统计测试集标签分布
test_label_stats = {
    "总样本数": len(test_df),
    "唯一值数量": test_df['label'].nunique(),
    "类别分布": test_df['label'].value_counts().to_dict(),
    "类别比例": test_df['label'].value_counts(normalize=True).to_dict()
}

# 打印结果
print("训练集标签统计:")
for k, v in train_label_stats.items():
    print(f"{k}: {v}")

print("测试集标签统计:")
for k, v in test_label_stats.items():
    print(f"{k}: {v}")

#
print(train_df.head())
print(train_df.describe())
# 保存结果
train_df.to_csv('../h_data/train.csv', index=False,header=False)
test_df.to_csv('../h_data/test.csv', index=False, header=False)