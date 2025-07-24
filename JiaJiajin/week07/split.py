import random
'''切割训练集和验证集'''
def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    num_lines = len(lines)
    num_train = int(0.8 * num_lines)

    train_lines = lines[:num_train]
    predict_lines = lines[num_train:]

    with open('data/train_data.csv', 'w', encoding='utf8') as train_data:
        train_data.writelines(train_lines)

    with open('data/predict_data.csv', 'w', encoding='utf8') as predict_data:
        predict_data.writelines(predict_lines)

split_file(r'H:/八斗网课/第七周 文本分类/week7 文本分类问题/文本分类练习.csv')
