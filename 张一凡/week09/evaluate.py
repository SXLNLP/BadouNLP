# -*- coding: utf-8 -*-
import json
import os
import torch
import numpy as np
from collections import defaultdict
from seqeval.metrics import f1_score, precision_score, recall_score
from loader import load_data
from transformers import BertTokenizer

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

        # 加载标签映射
        with open(config["schema_path"], encoding="utf-8") as f:
            self.label2id = json.load(f)
            if 'O' not in self.label2id:
                self.label2id['O'] = len(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}

        # 实体类型列表（自动从schema中提取）
        self.entities = list({
            label.split('-')[1] for label in self.label2id
            if '-' in label and label.split('-')[1] not in ['CLS', 'SEP']
        })

        # 加载验证数据
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.logger.info(f"验证集加载完成，样本数: {len(self.valid_data.dataset)}")

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.model.eval()

        true_labels, pred_labels = [], []
        sample_count = 0

        with torch.no_grad():
            for batch in self.valid_data:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.config["device"]),
                    "attention_mask": batch["attention_mask"].to(self.config["device"])
                }
                batch_labels = batch["labels"].cpu().numpy()

                # 模型预测
                if self.config["use_crf"]:
                    batch_preds = self.model(**inputs)
                else:
                    logits = self.model(**inputs)
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # 对齐标签
                for i in range(len(batch_labels)):
                    true = []
                    pred = []
                    for j in range(len(batch_labels[i])):
                        if batch_labels[i][j] != -1:  # 忽略padding
                            true.append(self.id2label.get(batch_labels[i][j], "O"))
                            pred.append(self.id2label.get(batch_preds[i][j], "O"))

                    true_labels.append(true)
                    pred_labels.append(pred)
                    sample_count += 1

                    # 打印前3个样本的预测结果
                    if sample_count <= 3:
                        words = self.tokenizer.convert_ids_to_tokens(
                            batch["input_ids"][i].cpu().numpy()
                        )
                        self.logger.info(f"样本{sample_count}预测结果:")
                        self.logger.info(f"文本: {' '.join(words)}")
                        self.logger.info(f"真实标签: {true}")
                        self.logger.info(f"预测标签: {pred}")
                        self.logger.info("--------------------")

        # 计算总体指标
        overall_metrics = {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels)
        }

        # 计算各实体指标
        entity_metrics = {}
        for entity in self.entities:
            # 提取特定实体的预测
            entity_true = [
                [l for l in sent if f"-{entity}" in l]
                for sent in true_labels
            ]
            entity_pred = [
                [l for l in sent if f"-{entity}" in l]
                for sent in pred_labels
            ]

            # 计算实体级指标
            tp = sum(1 for t, p in zip(entity_true, entity_pred) if t == p)
            fp = sum(1 for t, p in zip(entity_true, entity_pred) if t != p and len(p) > 0)
            fn = sum(1 for t, p in zip(entity_true, entity_pred) if t != p and len(t) > 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            entity_metrics[entity] = {
                "support": len([x for x in entity_true if x]),
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        # 打印结果
        self.logger.info("\n===== 评估结果 =====")
        self.logger.info(f"Overall - Precision: {overall_metrics['precision']:.4f}, "
                         f"Recall: {overall_metrics['recall']:.4f}, "
                         f"F1: {overall_metrics['f1']:.4f}")

        for entity, metrics in entity_metrics.items():
            self.logger.info(
                f"{entity} (support: {metrics['support']}) - "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}"
            )

        return overall_metrics['f1']

    def decode_predictions(self, preds, input_ids):
        """将预测ID转换为可读标签"""
        return [
            [self.id2label.get(p, "O") for p in pred if p != -1]
            for pred in preds
        ]
