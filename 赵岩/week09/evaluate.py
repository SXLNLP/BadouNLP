# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
import logging

"""
模型效果测试
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attention_mask, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id, attention_mask=attention_mask) # 输入变化时这里需要修改，比如多输入，多输出的情况
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

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
        schema_reverse = {
            0: "B-LOCATION",
            1: "B-ORGANIZATION",
            2: "B-PERSON",
            3: "B-TIME",
            4: "I-LOCATION",
            5: "I-ORGANIZATION",
            6: "I-PERSON",
            7: "I-TIME",
            8: "O"
        }
        entities = defaultdict(list)
        current_entity = ""
        entity_type = ""

        for i, label in enumerate(labels):
            if label == -1:  # Skip padding tokens
                continue
            if i >= len(sentence):  # Ensure we do not exceed the length of the sentence
                break
            tag = schema_reverse[label]
            if tag.startswith("B-"):
                if current_entity:
                    entities[entity_type].append(current_entity)
                current_entity = sentence[i]
                entity_type = tag[2:]
            elif tag.startswith("I-"):
                if current_entity and entity_type == tag[2:]:
                    current_entity += sentence[i]
                else:
                    if current_entity:
                        entities[entity_type].append(current_entity)
                    current_entity = ""
                    entity_type = ""
            else:
                if current_entity:
                    entities[entity_type].append(current_entity)
                current_entity = ""
                entity_type = ""

        if current_entity:
            entities[entity_type].append(current_entity)

        results = {
            "LOCATION": [],
            "TIME": [],
            "PERSON": [],
            "ORGANIZATION": []
        }

        for entity_type, ents in entities.items():
            if entity_type in ["LOCATION", "TIME", "PERSON", "ORGANIZATION"]:
                results[entity_type] = ents

        return results



