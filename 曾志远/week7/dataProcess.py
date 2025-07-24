import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from config import Config
from performance import Performance

"""
处理数据  将数据分为训练集和测试集
"""


class DataProcessor:
    def __init__(self, inputPath, text_column, label_column, proportion: float):  # proportion 训练集占所有数据的比例
        self.inputPath = inputPath
        self.text_column = text_column
        self.label_column = label_column
        self.proportion = proportion
        self.optData()

    def optData(self):
        # 确保输出路径存在  不存在则新增该目录
        # os.makedirs(self.outputPath, exist_ok=True)
        # 读取Excel文件
        df = pd.read_csv(self.inputPath)
        # 检查列是否存在
        if self.text_column not in df.columns or self.label_column not in df.columns:
            missing = [col in df.columns for col in [self.text_column, self.label_column] if col not in df.columns]
            raise ValueError(f'Missing columns: {missing}')

        # 提取文本及标签数据
        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()

        # 按proportion比例分割数据集 (随机打乱，固定随机种子确保可复现)
        X_train, X_test, Y_train, Y_test = train_test_split(
            texts,
            labels,
            train_size=self.proportion,
            random_state=40,  # 固定随机种子
            shuffle=True
        )

        # 将数据转为字典类型
        train_data = [{'text': x_train, 'label': y_train} for x_train, y_train in zip(X_train, Y_train)]
        train_label_1 = len([y for y in Y_train if y == 1])
        train_label_0 = len([y for y in Y_train if y == 0])
        train_sum_sentence_length = sum(len(x) for x in X_train)
        print("训练集总数：%d,正样本数：%d,负样本数：%d" % (len(train_data), train_label_1, train_label_0))
        print("训练集平均文本长度：%f" % (train_sum_sentence_length/len(X_train)))
        test_data = [{'text': x_test, 'label': y_text} for x_test, y_text in zip(X_test, Y_test)]
        test_label_1 = len([y for y in Y_test if y == 1])
        test_label_0 = len([y for y in Y_test if y == 0])
        test_sum_sentence_length = sum(len(x) for x in X_test)
        print("测试集总数：%d,正样本数：%d,负样本数：%d" % (len(test_data), test_label_1, test_label_0))
        print("训练集平均文本长度：%f" % (test_sum_sentence_length / len(X_test)))

        # 构建输出路径
        train_path = os.path.join(Config["train_data_path"])
        test_path = os.path.join(Config["valid_data_path"])

        # 保存为JSON文件
        with open(train_path, 'w', encoding='utf-8') as f:
            for text, label in zip(X_train, Y_train):
                # 创建单个JSON对象
                json_obj = {"label": label, "text": text}
                # 转换为JSON字符串并写入，末尾加换行符
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

        with open(test_path, 'w', encoding='utf-8') as f:
            for text, label in zip(X_test, Y_test):
                # 创建单个JSON对象
                json_obj = {"label": label, "text": text}
                # 转换为JSON字符串并写入，末尾加换行符
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

        return train_data, test_data


# 使用示例
if __name__ == "__main__":
    DataProcessor("./data/文本分类练习.csv", "review", "label", 0.8)
