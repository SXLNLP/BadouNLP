# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试 - 修改为适配BERT
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.id2label = {v: k for k, v in self.valid_data.dataset.schema.items()}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch in enumerate(self.valid_data):
            batch = {k: v.to(self.config["device"]) for k, v in batch.items()}
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]

            with torch.no_grad():
                outputs = self.model(**batch)

            pred_results = outputs.logits.argmax(dim=-1)
            self.write_stats(batch["labels"], pred_results, sentences)

        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            # 移除padding和特殊token
            active_tokens = (true_label != -100)
            true_label = true_label[active_tokens].cpu().tolist()
            pred_label = pred_label[active_tokens].cpu().tolist()

            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
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
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    def decode(self, sentence, label_ids):
        results = defaultdict(list)
        current_entity = ""
        current_label = None

        for i, label_id in enumerate(label_ids):
            if label_id == -100:  # 忽略特殊token
                continue

            label = self.id2label.get(label_id, "O")

            if label.startswith("B-"):
                if current_entity:  # 保存上一个实体
                    results[current_label].append(current_entity)
                current_entity = sentence[i]
                current_label = label[2:]
            elif label.startswith("I-"):
                if current_label == label[2:]:  # 同一实体继续
                    current_entity += sentence[i]
                else:  # 不匹配的I标签当作新实体开始
                    if current_entity:
                        results[current_label].append(current_entity)
                    current_entity = sentence[i]
                    current_label = label[2:]
            else:  # O
                if current_entity:
                    results[current_label].append(current_entity)
                current_entity = ""
                current_label = None

        if current_entity:  # 添加最后一个实体
            results[current_label].append(current_entity)

        return results
