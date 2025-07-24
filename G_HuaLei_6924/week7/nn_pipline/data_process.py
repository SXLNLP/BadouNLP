import csv
import json
import random


def buildData():
    # 读取CSV文件并转换为字典列表
    csv_file = '../文本分类练习数据集/文本分类练习.csv'
    train_json_file = '../week7_text_csv/train_consumption_review.json'
    valid_json_file = '../week7_text_csv/valid_consumption_review.json'

    train_data = []
    valid_data = []
    a = 0
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)  # 自动将表头作为键
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue
            append_data = {'tag': row[0], 'title': row[1]}
            if random.random() > 0.1:
                train_data.append(append_data)
            else:
                valid_data.append(append_data)

    # 转换为 NDJSON
    with open(train_json_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        # json.dump(train_data, f, indent=4, ensure_ascii=False)  # indent美化格式，ensure_ascii支持中文



    with open(valid_json_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        # json.dump(valid_data, f, indent=4, ensure_ascii=False)  # indent美化格式，ensure_ascii支持中文


# # 方法1：逐行读取（列表形式）
# with open('../文本分类练习数据集/train_consumption_review.json', 'r', encoding='utf-8') as file:
#     reader = csv.reader(file)  # 默认逗号分隔
#     for row in reader:
#         print('reader: ', row)  # 每行是一个列表，如 ['Name', 'Age', 'City']
#
# # 方法2：读取为字典（表头作为键）
# with open('../文本分类练习数据集/文本分类练习.csv', 'r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         print('DictReader: ', row)  # 每行是一个字典，如 {'Name': 'Tom', 'Age': '20', 'City': 'Beijing'}