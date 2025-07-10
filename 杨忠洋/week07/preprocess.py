# preprocess.py
import csv
import json
import random
import re
from collections import defaultdict

# 在split_data函数中添加过采样
from imblearn.over_sampling import RandomOverSampler

from config import Config


# 1. 数据加载与清洗
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 移除特殊字符
            review = re.sub(r'[^\w\s]', '', row['review'])
            data.append({
                'label': int(row['label']),
                'review': review
            })
    return data


# 2. 数据分析
def analyze_data(data):
    # 正负样本统计
    label_count = defaultdict(int)
    total_length = 0

    for item in data:
        label_count[item['label']] += 1
        total_length += len(item['review'])

    print(f"正样本数(好评): {label_count[1]}")
    print(f"负样本数(差评): {label_count[0]}")
    print(f"平均文本长度: {total_length / len(data):.2f}字符")

    return label_count


# 3. 数据集划分
def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


# 4. 保存为JSON格式
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps({
                "title": item['review'],
                "tag": "好评" if item['label'] == 1 else "差评"
            }, ensure_ascii=False)
            f.write(json_line + '\n')


def balance_data(data):
    features = [d['review'] for d in data]
    labels = [d['label'] for d in data]

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(np.array(features).reshape(-1, 1), labels)

    balanced_data = []
    for i in range(len(X_res)):
        balanced_data.append({
            'review': X_res[i][0],
            'label': y_res[i]
        })
    return balanced_data


if __name__ == "__main__":
    # 加载原始数据
    raw_data = load_data(Config["raw_data_path"])
    raw_data = balance_data(raw_data)
    # 数据分析
    analyze_data(raw_data)

    # 划分数据集
    train_data, valid_data = split_data(raw_data)

    # 保存数据集
    save_json(train_data, Config["train_data_path"])
    save_json(valid_data, Config["valid_data_path"])

    print("数据集预处理完成!")