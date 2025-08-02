# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
import json
from collections import defaultdict
from loader import load_data
from transformers import BertTokenizer

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        
        for index, batch in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask, token_type_ids)
            
            self.write_stats(labels.cpu().numpy(), pred_results, sentences)
        
        self.show_stats()
        return

    def write_stats(self, true_labels, pred_labels, sentences):
        for i in range(len(sentences)):
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            sentence = sentences[i]
            
            # 获取有效标签索引
            mask = true_label != -100
            valid_true_label = true_label[mask]

            # 获取有效位置数量（n个有效token）
            n_valid = mask.sum()

            # 只保留前 n 个预测结果（避免长度不一致）
            valid_pred_label = np.array(pred_label)[:n_valid]

            
            true_entities = self.decode(sentence, valid_true_label)
            pred_entities = self.decode(sentence, valid_pred_label)
            
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in self.stats_dict])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in self.stats_dict])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in self.stats_dict])
        
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels])
        results = defaultdict(list)
        
        # 使用BIO规则解码
        current_entity = ""
        current_type = ""
        start_index = -1
        
        for i, label in enumerate(labels):
            # 实体开始
            if label in ["0", "1", "2", "3"]:  # B-标签
                if current_entity:  # 保存上一个实体
                    results[current_type].append(current_entity)
                current_type = self.get_entity_type(label)
                current_entity = sentence[i]
                start_index = i
            
            # 实体内部
            elif label in ["4", "5", "6", "7"] and current_entity:  # I-标签
                # 确保I标签类型匹配
                if self.get_entity_type(label) == current_type:
                    current_entity += sentence[i]
            
            # 实体结束
            else:
                if current_entity:
                    results[current_type].append(current_entity)
                    current_entity = ""
                    current_type = ""
        
        # 处理最后一个实体
        if current_entity:
            results[current_type].append(current_entity)
        
        return results

    def get_entity_type(self, label):
        mapping = {
            "0": "LOCATION",
            "1": "ORGANIZATION",
            "2": "PERSON",
            "3": "TIME",
            "4": "LOCATION",
            "5": "ORGANIZATION",
            "6": "PERSON",
            "7": "TIME"
        }
        return mapping.get(label, "")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
