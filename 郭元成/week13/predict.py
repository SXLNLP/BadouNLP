# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer, BertModel
from peft import get_peft_model, LoraConfig
from main import peft_wrapper

"""
模型效果测试
"""


class NER:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = self.load_vocab(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        model = TorchModel(config)
        model = peft_wrapper(model)
        state_dict = model.state_dict()
        state_dict.update(torch.load(model_path))
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        return BertTokenizer.from_pretrained(vocab_path)

    def encode_sentence(self, text, padding=True):
        return self.tokenizer.encode(text,
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)

    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(int(x)) for x in labels[:len(sentence) + 1]])
        results = defaultdict(list)

        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

    def predict(self, sentence):
        input_ids = self.encode_sentence(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_ids]))[0]
            labels = torch.argmax(res, dim=-1)
        results = self.decode(sentence, labels)
        return results


if __name__ == "__main__":
    sl = NER(Config, "model_output/epoch_5.pth")
    sentence = "(本报约翰内斯堡电)本报记者安洋贺广华留学人员档案库建立本报讯中国质量体系认证机构国家认可委员会日前正式签署了国际上第一个质量认证的多边互认协议,表明中国质量体系认证达到了国际水平。"
    res = sl.predict(sentence)
    print(res)