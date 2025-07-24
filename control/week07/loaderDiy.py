#encoding:utf-8

import torch
from transformers import BertTokenizer
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader

class DataGeneratorDiy:
    def __init__(self, config):
        self.config = config
        self.dataset_path = self.config["dataset_path"]
        self.train_data_percent = self.config["train_data_percent"]
        self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        names = ['label', 'review']
        dataset = read_csv(self.dataset_path, names=names, skiprows=1)  # 读取csv文件，指定列明，且跳过第一行，默认第一行是列名
        print(dataset.shape)
        train_data_set = dataset.sample(frac=self.train_data_percent, random_state=0)
        test_data_test = dataset.drop(train_data_set.index)
        self.train_data = self.__wordembedding(train_data_set)
        self.test_data = self.__wordembedding(test_data_test)
        self.train_data_distribute = self.calcute_percent(dataset)
        self.test_data_distribute = self.calcute_percent(test_data_test)
        # 训练，测试集中，正反例的分布情况
        self.config["train_data_distribute"] =  self.train_data_distribute
        self.config["test_data_distribute"] = self.test_data_distribute

        return

    def __wordembedding(self, datas):
        data = []
        for index, row in datas.iterrows():
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(row["review"], max_length=self.config["max_length"], pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(row["review"])
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([row["label"]])
            data.append([input_id, label_index])
        return data

    def calcute_percent(self, datas):
        positive_num = datas['label'].sum() # 集合中，正例的个数
        negative_num = datas.shape[0] - positive_num
        return (positive_num,negative_num)


    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        return self.train_data[index]

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_data_distribute(self):
        return self.train_data_distribute

    def get_test_data_distribute(self):
        return self.test_data_distribute

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

def load_data(datas, config ,shuffle=True):
    dl = DataLoader(datas , batch_size=config["batch_size"], shuffle=shuffle)
    return dl

#用torch自带的DataLoader类封装数据
if __name__ == "__main__":
    from config import Config
    dg = DataGeneratorDiy(Config)
    print(dg[1])
