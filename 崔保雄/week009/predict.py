import os

import jieba
import torch
from config import Config
from model import TorchModel
from transformers import BertTokenizer

#文本标注模型预测
class Predict():
    def __init__(self):
        self.config = Config
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_path"])
        self.use_bert = False
        if self.config["model_type"] == "bert":
            self.use_bert = True


    """ 模型预测的方法 """
    def eval(self, text):

        """ 第1步：将句子整理成向量，要注意区分是基于bert，还是基于lstm """
        input_id = self.encode_sentence(text, padding=True)
        input_id = torch.LongTensor([input_id])

        """ 第2步：将句子整理成向量，要注意区分是基于bert，还是基于lstm """
        model = TorchModel(self.config)
        model_path = os.path.join(self.config["model_path"], f"ner_{self.config["model_type"]}.pth")
        model.load_state_dict(torch.load(model_path))

        """ 第3步：将向量传到模型预测 """
        #这里使用评估/测试模式
        model.eval()
        with torch.no_grad():
            y = model.forward(input_id)
        return {"input": text, "output": y}


    def encode_sentence(self, text, padding=True):
        input_id = []

        if self.use_bert:
            """ 使用bert编码，其实用的是bert自己的词表 """
            encodes = self.tokenizer.encode(text)
            for encode in encodes:
                input_id.append(encode)
        else:
            """ 使用lstm，就得使用自定义词表 """
            if self.config["vocab_path"] == "words.txt":
                for word in jieba.cut(text):
                    input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
            else:
                for char in text:
                    input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))

        if padding:
            input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        return token_dict
