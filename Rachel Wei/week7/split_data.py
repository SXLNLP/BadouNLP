import csv
import random
from config import Config
from sklearn.model_selection import train_test_split

def split_data(Config):
    input_file = Config['entire_data_path']
    train_file = Config['train_data_path']
    valid_file = Config['valid_data_path']
    Seed = Config['seed']

    with open(input_file, encoding="utf8") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        rows = reader[1:]

    train_rows, valid_rows = train_test_split(rows, test_size = 0.2, random_state=Seed)

    ## Save Files
    with open(train_file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    with open(valid_file, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_rows)

    print('successfully split data into train vs. test datasets')