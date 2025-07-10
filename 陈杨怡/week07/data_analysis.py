import pandas as pd

df = pd.read_csv('data/文本分类练习.csv')
pos = df[df['label'] == 1].shape[0]
neg = df[df['label'] == 0].shape[0]
avg_len = df['text'].astype(str).apply(len).mean()

print(f"正样本数: {pos}")
print(f"负样本数: {neg}")
print(f"文本平均长度: {avg_len:.2f}")
