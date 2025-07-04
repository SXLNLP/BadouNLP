import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import sympy

# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    # 加载CSV文件
    df = pd.read_csv(file_path, encoding='utf-8')

    # 清洗文本：保留中文字符，删除标点
    def clean_text(text):
        text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)  # 仅保留中文字符
        return text.strip()

    # 中文分词（使用jieba）
    def preprocess(text):
        return ' '.join(jieba.cut(clean_text(text)))

    df['processed_text'] = df['review'].apply(preprocess)
    return df


# 2. 数据分析（样本数、文本长度等）
def analyze_data(df):
    print("=== 数据统计 ===")
    print("正负样本分布:")
    print(df['label'].value_counts())

    df['text_len'] = df['review'].str.len()
    print("\n文本平均长度:")
    print(df.groupby('label')['text_len'].mean())

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='text_len', hue='label', bins=50, kde=True)
    plt.title('文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    plt.legend(title='类别', labels=['差评 (0)', '好评 (1)'])
    plt.show()


# 3. 模型训练与评估
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_val)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} 训练完成：准确率 {accuracy:.4f}, 训练时间 {train_time:.2f}s, 预测时间 {predict_time:.4f}s")
    return model, accuracy, train_time, predict_time


def main(file_path):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 数据预处理
    df = load_and_preprocess_data(file_path)
    analyze_data(df)

    # 数据划分
    X_train, X_val, y_train, y_val = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # TF-IDF特征提取（中文文本）
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    # 模型定义
    models = {
        "朴素贝叶斯": MultinomialNB(),
        "逻辑回归": LogisticRegression(solver='liblinear'),
        "随机森林": RandomForestClassifier(n_estimators=100),
    }

    results = []

    # 传统模型训练
    for model_name, model in models.items():
        model, accuracy, train_time, predict_time = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val,
                                                                   model_name)
        results.append({
            '模型': model_name,
            '准确率': accuracy,
            '训练时间(s)': train_time,
            '预测时间(s)': predict_time
        })

    # 深度学习模型（PyTorch实现）
    print("\n=== 训练深度学习模型（PyTorch） ===")

    class TextClassifier(nn.Module):
        def __init__(self, input_dim=5000):
            super(TextClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # 设备选择（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)

    # 移动到设备
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_val_tensor, y_val_tensor = X_val_tensor.to(device), y_val_tensor.to(device)

    # 实例化模型、优化器、损失函数
    model = TextClassifier(input_dim=5000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    # 开始训练
    start_train_time = time.time()
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_train_time

    # 模型预测
    start_predict_time = time.time()
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        y_pred = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
    predict_time = time.time() - start_predict_time

    accuracy = accuracy_score(y_val.to_numpy(), y_pred)

    results.append({
        '模型': '深度神经网络(DNN-PyTorch)',
        '准确率': accuracy,
        '训练时间(s)': train_time,
        '预测时间(s)': predict_time
    })

    # 结果展示
    results_df = pd.DataFrame(results)
    print("\n=== 结果对比表格 ===")
    print(results_df.to_string(index=False))

    # 可视化准确率对比
    plt.figure(figsize=(10, 5))
    sns.barplot(x='模型', y='准确率', data=results_df)
    plt.title('PyTorch 模型准确率对比')
    plt.ylim(0.4, 1.0)
    plt.xticks(rotation=45)
    plt.show()


# 执行主函数
if __name__ == "__main__":
    file_path = "文本分类练习.csv"
    main(file_path)
