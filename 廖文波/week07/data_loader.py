
from transformers import BertTokenizer
import json
import torch
import random

class DataLoader:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0:'差评', 1:'好评'}  # 标签索引映射
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()  # 加载数据
    
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"]
                label = self.label_to_index[tag]
                # print("当前标签", tag, "对应索引", label)
                text = line["text"]
                if self.config["model_type"] == "bert":
                    # print("embding前内容",text)
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
                    # print("embding后内容",input_id)
                else:
                    input_id = self.encode_sentence(text)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return
    
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

'''
根据原始数据构建训练数据与验证数据
'''
def generate_data(data_path):
    positives_data = [] ## 正样本数据
    negatives_data = [] ## 负样本数据
    positives_train_data_ratio = 0.9  # 正样本训练集比例
    negatives_train_data_ratio = 0.5  # 负样本训练集比例
    # 读取数据,并进行正负样本分类
    with open(data_path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",",1)
            if("1" == parts[0]):
                result_dict = {"label":'好评', "text":parts[1]}
                positives_data.append(result_dict)
            elif("0" == parts[0]):
                result_dict = {"label":'差评', "text":parts[1]}
                negatives_data.append(result_dict)
   
    print("正样本数：", len(positives_data), "负样本数：", len(negatives_data))
    #随机抽取80%的正样本与80%的负样本作为训练集，并交叉混合正负样本数据
    positives_len = int(positives_train_data_ratio * len(positives_data))
    print("正样本数：", positives_len)
    negatives_len = int(negatives_train_data_ratio * len(negatives_data))
    print("负样本数：", negatives_len)
    train_data = positives_data[:positives_len] + negatives_data[:negatives_len]
    #将正负样本数据打乱顺序
    random.shuffle(train_data)
    #随机抽取20%的正样本与20%的负样本作为验证集，并交叉混合正负样本数据
    test_data = positives_data[positives_len:] + negatives_data[negatives_len:]
    #将正负样本数据打乱顺序
    random.shuffle(test_data)
    #将两份数据循环按行以json文件格式写入当前路径下basedata文件夹
    with open("week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/train_data.json", "w", encoding="utf8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/valid_data.json", "w", encoding="utf8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_data(data_path, config, shuffle=True):
    """
    用torch自带的DataLoader类封装数据
    """
    dl = DataLoader(data_path, config)
    return torch.utils.data.DataLoader(dl, batch_size=config["batch_size"], shuffle=shuffle)

if __name__ == "__main__":
    #生成数据
    # generate_data("week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/文本分类练习.csv")
    from config import Config
    dl = load_data("week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/valid_data.json", Config)
    print(dl.dataset.data)  # 打印第一条数据