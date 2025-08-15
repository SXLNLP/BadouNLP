# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from peft import get_peft_model, LoraConfig
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)

        # 检查是否是LoRA模型
        if "lora_ner" in model_path and config.get("tuning_tactics") == "lora_tuning":
            # 配置并应用LoRA
            peft_config = LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["lora_target_modules"]
            )
            self.model.bert = get_peft_model(self.model.bert, peft_config)

            # 加载LoRA参数
            saved_params = torch.load(model_path, map_location='cpu')
            # 只加载与当前模型匹配的参数
            model_dict = self.model.state_dict()
            filtered_params = {k: v for k, v in saved_params.items() if k in model_dict}
            model_dict.update(filtered_params)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"LoRA模型加载完毕! 加载了 {len(filtered_params)} 个参数")
        else:
            # 标准模型加载
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print("标准模型加载完毕!")

        self.model.eval()

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表 - 与loader.py保持一致
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))

        # 确保输入长度不超过max_length
        if len(input_id) > self.config["max_length"]:
            input_id = input_id[:self.config["max_length"]]

        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            if not self.config["use_crf"]:
                res = torch.argmax(res, dim=-1)
            if hasattr(res, 'cpu'):
                res = res.cpu().detach().tolist()
            elif not isinstance(res, list):
                res = list(res)

        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

    def extract_entities(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))

        # 确保输入长度不超过max_length
        if len(input_id) > self.config["max_length"]:
            input_id = input_id[:self.config["max_length"]]
            sentence = sentence[:self.config["max_length"]]

        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            if not self.config["use_crf"]:
                res = torch.argmax(res, dim=-1)
            if hasattr(res, 'cpu'):
                res = res.cpu().detach().tolist()
            elif not isinstance(res, list):
                res = list(res)

        return self.decode(sentence, res)

    def decode(self, sentence, labels):
        if not isinstance(labels, list):
            labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        labels_str = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)

        for match in re.finditer("(04*)", labels_str):
            s, e = match.span()
            results["LOCATION"].append(sentence[s:e])

        for match in re.finditer("(15*)", labels_str):
            s, e = match.span()
            results["ORGANIZATION"].append(sentence[s:e])

        for match in re.finditer("(26*)", labels_str):
            s, e = match.span()
            results["PERSON"].append(sentence[s:e])

        for match in re.finditer("(37*)", labels_str):
            s, e = match.span()
            results["TIME"].append(sentence[s:e])

        return results

if __name__ == "__main__":
    # 根据配置选择模型文件
    if Config.get("tuning_tactics") == "lora_tuning":
        model_path = "./model_output/lora_ner_epoch_20.pth"
    else:
        model_path = "./model_output/epoch_20.pth"

    sl = SentenceLabel(Config, model_path)

    sentence = "王励勤、阎森、马琳在北京举行的乒乓球比赛中获得冠军"

    # 添加调试信息
    input_id = []
    for char in sentence:
        input_id.append(sl.vocab.get(char, sl.vocab["[UNK]"]))

    with torch.no_grad():
        raw_output = sl.model(torch.LongTensor([input_id]))[0]
        print(f"原始模型输出前10个: {raw_output[:10]}")

        if not sl.config["use_crf"]:
            res = torch.argmax(raw_output, dim=-1)
        else:
            res = raw_output

        if hasattr(res, 'cpu'):
            res = res.cpu().detach().tolist()
        elif not isinstance(res, list):
            res = list(res)

        print(f"预测标签: {res[:len(sentence)]}")

    res = sl.predict(sentence)
    print(f"标注结果: {res}")

    entities = sl.extract_entities(sentence)
    print(f"识别实体: {entities}")
    print()

    sentence = "在邓小平同志逝世一周年的时候,我们在中国北京缅怀他"
    res = sl.predict(sentence)
    print(f"标注结果: {res}")

    entities = sl.extract_entities(sentence)
    print(f"识别实体: {entities}")
    print()

    sentence = "中共中央政治局委员、书记处书记丁关根今天主持座谈会"
    res = sl.predict(sentence)
    print(f"标注结果: {res}")

    entities = sl.extract_entities(sentence)
    print(f"识别实体: {entities}")