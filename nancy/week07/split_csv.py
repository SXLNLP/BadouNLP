import pandas as pd

# 读取原始数据
df = pd.read_csv("文本分类练习.csv")

# 按8:2拆分
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# 保存为新文件
train_df.to_csv("train.csv", index=False)
test_df.to_csv("valid.csv", index=False)
