# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from transformers import BertTokenizer
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for batch_data in self.valid_data:
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_logits = outputs.logits
            pred_labels = torch.argmax(pred_logits, dim=-1)  # 获取预测标签
            self.write_stats(labels, pred_labels, input_ids)
        self.show_stats()
        return

    def write_stats(self, labels, pred_labels, input_ids):
        assert len(labels) == len(pred_labels) == len(input_ids)

        for true_label, pred_label, input_id in zip(labels, pred_labels, input_ids):
            input_id = input_id.cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            
            # 计算有效长度（去除padding）
            seq_len = len([t for t in input_id if t != self.tokenizer.pad_token_id])
            
            # 截取有效部分
            input_id = input_id[:seq_len]
            tokens = tokens[:seq_len]
            true_label = true_label[:seq_len].cpu().tolist()
            pred_label = pred_label[:seq_len].cpu().tolist()

            # 过滤掉标签为-100的位置
            valid_indices = [i for i, label in enumerate(true_label) if label != -100]
            valid_true = [true_label[i] for i in valid_indices]
            valid_pred = [pred_label[i] for i in valid_indices]
            valid_tokens = [tokens[i] for i in valid_indices]

            # 重构句子
            sentence = self.tokenizer.convert_tokens_to_string(valid_tokens).replace(" ", "")
            
            # 解码实体
            true_entities = self.decode(sentence, valid_true)
            pred_entities = self.decode(sentence, valid_pred)

            # 更新统计信息
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
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


