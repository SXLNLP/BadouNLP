# 先处理数据，划分训练集和测试集，并把csv数据转换成json

import pandas as pd
import json
from sklearn.model_selection import train_test_split

def pre_process(data_path):
    f = pd.read_csv(data_path)
    data_dicts = [{"label":int(row["label"]), "review":row["review"]} for index, row in f.iterrows()]
    train_data, test_data = train_test_split(data_dicts, test_size=0.1, random_state=42, stratify=[d['label'] for d in data_dicts])
    with open('../data/train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open('../data/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(len(train_data))
    print(len(test_data))
    return train_data, test_data

def max_length(train_data, test_data):
    review_len1 = [len(i['review']) for i in train_data]
    review_len2 = [len(i['review']) for i in test_data]
    print(pd.Series(review_len1).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    print(pd.Series(review_len2).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

def main(path):
    train_data,test_data = pre_process(path)
    max_length(train_data, test_data)

if __name__ == '__main__':
    main(r'..\data\文本分类练习.csv')
    # 得到结果
    # train_data:mean        24.942807
    #            95%         69.000000
    #            99%         125.000000
    # test_data:mean         26.000000
    #           95%          72.000000
    #           99%          132.040000
    # 选取max_length = 30, 72