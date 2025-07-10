# main.py
import numpy as np
import torch
import pandas as pd
from config_homework import Config
from loader import load_dataset, TextDataset, tokenizer
from torch.utils.data import DataLoader
from evaluate import train_eval

from sklearn.model_selection import train_test_split

# 设置随机种子
torch.manual_seed(Config["seed"])
np.random.seed(Config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
df = load_dataset()
print(f"总样本数：{len(df)}")
print(f"正样本：{df['label'].sum()}，负样本：{len(df) - df['label'].sum()}")
print(f"平均文本长度：{df['text'].apply(lambda x: len(str(x))).mean():.2f}")

# 分割数据
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=Config["seed"])

# 构造数据集
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=Config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 运行
results = []
for mode in ['dnn', 'cnn', 'rnn']:
    print(f"\n 正在训练模型结构：{mode.upper()}")
    acc, pred_time = train_eval(mode, train_loader, val_loader, df)
    results.append({
        "Model": f"BERT+{mode.upper()}",
        "Learning_Rate": Config["learning_rate"],
        "Hidden_Size": Config["hidden_size"],
        "acc": f"{acc:.4f}",
        "time(预测100条耗时)": f"{pred_time:.2f}s"
    })

df_result = pd.DataFrame(results)
print("\n 实验结果对比表格：")
print(df_result.to_string(index=False))
df_result.to_csv("模型比较结果.csv", index=False)
