import pandas as pd
from sklearn.model_selection import train_test_split

"""
把数据集分为训练集和验证集
"""
df = pd.read_csv("文本分类练习.csv")
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)