# evaluate.py
# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
import json
from collections import defaultdict
from loader import load_data

"""
模型效果测试（适配BERT+CRF+schema.json）
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.schema = self.load_schema(config["schema_path"])
        self.label_id_map = self.build_label_map(self.schema)  # {"B-LOC": 0, ...}
        self.entity_map = self.build_entity_pattern(self.schema)  # {"LOCATION": ["04+"], ...}
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.device = next(model.parameters()).device

    def load_schema(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_label_map(self, schema):
        return {v: k for k, v in schema.items()}

    def build_entity_pattern(self, schema):
        result = defaultdict(list)
        for label, idx in schema.items():
            if label == "O":
                continue
            if label.startswith("B-"):
                ent_type = label[2:]
                i_label = f"I-{ent_type}"
                b_id = idx
                i_id = schema.get(i_label)
                if i_id is not None:
                    result[ent_type.upper()].append(f"{b_id}{i_id}+")
        return result

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {key: defaultdict(int) for key in self.entity_map}

        self.model.eval()
        for batch_data in self.valid_data:
            if self.config.get("use_bert", False):
                input_ids, labels, attention_mask, sentences = batch_data
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device).bool()
                with torch.no_grad():
                    pred_results = self.model(input_ids, attention_mask=attention_mask)
            else:
                input_ids, labels, sentences = batch_data
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    pred_results = self.model(input_ids)

            self.write_stats(labels, pred_results, sentences)

        self.show_stats()

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)

        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1).cpu().tolist()
        else:
            if isinstance(pred_results, torch.Tensor):
                pred_results = pred_results.cpu().tolist()

        for i in range(len(labels)):
            true_label = labels[i].cpu().tolist()
            pred_label = pred_results[i]
            sentence = sentences[i]

            # 只保留非 -100 的标签（subword部分是 -100）
            clean_true, clean_pred = [], []
            for tl, pl in zip(true_label, pred_label):
                if tl != -100:
                    clean_true.append(tl)
                    clean_pred.append(pl)

            true_entities = self.decode(sentence, clean_true)
            pred_entities = self.decode(sentence, clean_pred)

            for key in self.stats_dict:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def show_stats(self):
        F1_scores = []
        for key in self.stats_dict:
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

    def decode(self, sentence, labels):
        """
        依据BIO模式和数字标签，构造实体字典
        """
        from collections import defaultdict
        results = defaultdict(list)
        entity = []
        entity_type = None

        for idx, label in enumerate(labels):
            if label == 8:  # O标签，结束当前实体
                if entity:
                    results[entity_type].append("".join(entity))
                    entity = []
                    entity_type = None
            elif label in [0, 1, 2, 3]:  # B-标签，开始新实体
                if entity:
                    results[entity_type].append("".join(entity))
                entity = [sentence[idx]]
                if label == 0:
                    entity_type = "LOCATION"
                elif label == 1:
                    entity_type = "ORGANIZATION"
                elif label == 2:
                    entity_type = "PERSON"
                elif label == 3:
                    entity_type = "TIME"
            elif label in [4, 5, 6, 7]:  # I-标签，继续当前实体
                if entity:
                    entity.append(sentence[idx])
                else:
                    # 遇到非法I标签，单独作为B处理
                    entity = [sentence[idx]]
                    if label == 4:
                        entity_type = "LOCATION"
                    elif label == 5:
                        entity_type = "ORGANIZATION"
                    elif label == 6:
                        entity_type = "PERSON"
                    elif label == 7:
                        entity_type = "TIME"
            else:
                # 标签异常，直接结束当前实体
                if entity:
                    results[entity_type].append("".join(entity))
                    entity = []
                    entity_type = None

        # 收尾
        if entity:
            results[entity_type].append("".join(entity))

        # 保证所有实体类型都有键，避免KeyError
        for ent_type in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]:
            if ent_type not in results:
                results[ent_type] = []

        return results
