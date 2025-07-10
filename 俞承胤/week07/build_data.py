import os.path

from config import Config
import csv
"""
训练数据预处理

"""

class build_data:
    def __init__(self):
        pass

if __name__ == "__main__":
    print(Config["data_file_path"])
    i = 0;positive = 0;negative = 0;textLength = 0

    newReview = ""
    if os.path.exists("train_data.csv"):
        os.remove("train_data.csv")
    if os.path.exists("verify_data.csv"):
        os.remove("verify_data.csv")
    with open(Config["data_file_path"], 'r', encoding='utf-8') as f,\
        open('train_data.csv', 'w', newline='', encoding='utf-8') as t_file,\
        open('verify_data.csv', 'w', newline='', encoding='utf-8') as v_file:
        reader = csv.DictReader(f)
        writer_t = csv.writer(t_file)
        writer_t.writerow(['label', 'review'])  # 写入表头
        writer_v = csv.writer(v_file)
        writer_v.writerow(['label', 'review'])  # 写入表头
        for row in reader:
            # print(f"类型：{row['label']} 内容：{row['review']}")
            newReview = row['review']
            newReview = newReview.replace('"', '')
            textLength += len(newReview)
            if i % 10 == 0:
                if row['label'] == '1':
                    positive += 1
                else:
                    negative += 1
                writer_v.writerow([row['label'],newReview])
            else:
                if row['label'] == '1':
                    positive += 1
                else:
                    negative += 1
                writer_t.writerow([row['label'],newReview])  # 写入多行数据
            i += 1
        print(f"平均长度：{textLength/i}")
        print(f"正例：{positive} 负例：{negative}")