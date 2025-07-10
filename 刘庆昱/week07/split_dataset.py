import os
import random
from week07.homework_pipline.config import Config


def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]

    random.shuffle(lines)
    lines_num = len(lines)
    num_train = int(0.8 * lines_num)
    train_lines = lines[:num_train]
    valid_lines = lines[num_train:]

    os.makedirs(os.path.dirname(Config["train_data_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(Config["valid_data_path"]), exist_ok=True)

    with open(Config["train_data_path"], 'w', encoding='utf8') as train_set:
        train_set.writelines(train_lines)

    with open(Config["valid_data_path"], 'w', encoding='utf8') as valid_set:
        valid_set.writelines(valid_lines)


split_file(Config["dataset_path"])