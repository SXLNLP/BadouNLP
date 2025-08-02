# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试 - 适配BERT版本
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.id2label = {int(v): k for k, v in self.load_schema(config["schema_path"]).items()}

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # 获取原始句子
            batch_start = index * self.config["batch_size"]
            batch_end = min((index+1) * self.config["batch_size"], len(self.valid_data.dataset.sentences))
            sentences = self.valid_data.dataset.sentences[batch_start:batch_end]
            
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
                
            # 输入现在是三个元素：input_ids, attention_mask, labels
            input_ids, attention_mask, labels = batch_data
            
            with torch.no_grad():
                if self.config["use_crf"]:
                    # CRF模式返回最佳路径
                    pred_results = self.model(input_ids, attention_mask)
                else:
                    # 非CRF模式返回预测概率分布
                    logits = self.model(input_ids, attention_mask)
                    # 获取预测标签
                    active_mask = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, logits.size(-1))[active_mask]
                    pred_results = torch.argmax(active_logits, dim=-1)
            
            self.write_stats(input_ids, attention_mask, labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, input_ids, attention_mask, labels, pred_results, sentences):
        assert len(sentences) == len(labels)
        
        # 处理每个样本
        for i in range(len(sentences)):
            # 获取当前样本的相关数据
            sentence = sentences[i]
            label_seq = labels[i].cpu().detach().tolist()
            attn_mask = attention_mask[i].cpu().detach().tolist()
            
            # 处理预测结果
            if self.config["use_crf"]:
                # CRF模式返回的是整个batch的解码路径列表
                pred_seq = pred_results[i]
            else:
                # 非CRF模式需要重建标签序列
                start_idx = i * self.config["max_length"]
                end_idx = start_idx + self.config["max_length"]
                # 只取当前样本的有效部分
                pred_seq = pred_results[start_idx:end_idx].tolist()
                # 根据注意力掩码截断
                pred_seq = pred_seq[:sum(attn_mask)]
            
            # 解码真实标签和预测标签
            true_entities = self.decode(sentence, label_seq, attn_mask)
            pred_entities = self.decode(sentence, pred_seq, attn_mask)
            
            # 更新统计信息
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

    def decode(self, sentence, labels, attention_mask=None):
        """
        将标签序列解码为实体
        """
        # 只保留有效标签（根据注意力掩码）
        if attention_mask:
            valid_labels = []
            for i, label in enumerate(labels):
                if i >= len(attention_mask):
                    break
                if attention_mask[i] == 1:
                    valid_labels.append(label)
            labels = valid_labels
        else:
            # 如果没有提供注意力掩码，截断到句子长度
            labels = labels[:len(sentence)]
        
        # 将标签ID转换为标签字符串
        label_str = "".join([self.id2label.get(label, "O") for label in labels])
        
        results = defaultdict(list)
        
        # 使用更健壮的模式匹配
        patterns = {
            "LOCATION": r"B-LOCATION(I-LOCATION)*",
            "ORGANIZATION": r"B-ORGANIZATION(I-ORGANIZATION)*",
            "PERSON": r"B-PERSON(I-PERSON)*",
            "TIME": r"B-TIME(I-TIME)*"
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, label_str):
                start, end = match.span()
                # 计算实体在原始文本中的位置
                entity_text = sentence[start:end]
                # 添加到结果
                results[entity_type].append(entity_text)
        
        return results
