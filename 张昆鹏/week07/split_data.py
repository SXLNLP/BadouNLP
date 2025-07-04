import csv
import random

def split_dataset(input_path, train_path, test_path, split_ratio=0.8, seed=42):
    random.seed(seed)

    with open(input_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  
        data = list(reader)

    random.shuffle(data)

    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    with open(train_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_data)

    with open(test_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_data)

    print(f"训练集保存至 {train_path}，共 {len(train_data)} 条数据")
    print(f"测试集保存至 {test_path}，共 {len(test_data)} 条数据")

if __name__ == "__main__":
    data_path = r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\文本分类练习.csv"
    train_path = r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\train_data.csv"
    test_path = r"N:\八斗\上一期\第七周 文本分类\week7 文本分类问题\homework\test_data.csv"
    split_dataset(data_path, train_path, test_path)