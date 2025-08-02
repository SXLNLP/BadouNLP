import random
import pandas as pd

def split_file(file_path):
    # 使用pandas读取CSV文件
    df = pd.read_csv(file_path)
    
    # 转换为模型需要的格式：标签,文本内容
    lines = []
    for _, row in df.iterrows():
        lines.append(f"{row['label']},{row['review']}\n")
    
    random.shuffle(lines)
    num_lines = len(lines)
    num_train = int(0.8 * num_lines)

    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    with open('C:/Users/yanglab1/Desktop/1/BadouNLP/汪桓玉/week07/train_data.txt', 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)

    with open('C:/Users/yanglab1/Desktop/1/BadouNLP/汪桓玉/week07/valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)

split_file(r'C:/Users/yanglab1/Desktop/1/BadouNLP/汪桓玉/week07/文本分类练习.csv')