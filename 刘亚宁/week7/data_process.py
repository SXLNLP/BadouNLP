import csv
import random


def buildData():
    csv_file = '../文本分类问题/文本分类练习.csv'
    train_csv_file = r'D:\week7\homework\data\train.csv'
    valid_csv_file = r'D:\week7\homework\data\evaluate.csv'

    train_data = []
    valid_data = []
    a = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)  
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue
            append_data = {'tag': row[0], 'title': row[1]}
            if random.random() > 0.1:
                train_data.append(append_data)
            else:
                valid_data.append(append_data)

    with open(train_csv_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(csv.dumps(item, ensure_ascii=False) + '\n')



    with open(valid_csv_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(csv.dumps(item, ensure_ascii=False) + '\n')
