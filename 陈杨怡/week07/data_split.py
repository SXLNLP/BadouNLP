import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_csv('data/文本分类练习.csv')

# 假设label列为“label”，好评为1，差评为0
train, valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 保存为json行格式，便于你的loader读取
train.to_json('data/train_tag_news.json', orient='records', force_ascii=False, lines=True)
valid.to_json('data/valid_tag_news.json', orient='records', force_ascii=False, lines=True)

print(f"训练集: {len(train)}，验证集: {len(valid)}")
