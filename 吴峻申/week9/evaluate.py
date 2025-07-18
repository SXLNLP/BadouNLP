import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        device = next(self.model.parameters()).device

        for batch_idx, batch in enumerate(self.valid_data):
            start_idx = batch_idx * self.config["batch_size"]
            end_idx = min((batch_idx + 1) * self.config["batch_size"], len(self.valid_data.dataset.sentences))
            sentences = self.valid_data.dataset.sentences[start_idx:end_idx]

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].cpu().numpy()

            with torch.no_grad():
                preds = self.model(input_ids, attention_mask=attention_mask)

            if not self.config["use_crf"]:
                preds = preds.cpu().numpy()

            self.write_stats(labels, preds, sentences)

        self.show_stats()
        return

    def write_stats(self, true_labels, pred_labels, sentences):
        for i in range(len(sentences)):
            true = true_labels[i]
            pred = pred_labels[i]
            sentence = sentences[i]

            # 移除填充标签
            valid_indices = np.where(true != -100)[0]
            true = true[valid_indices]
            pred = pred[:len(true)]  # 确保预测长度与真实标签一致

            true_entities = self.decode(sentence, true)
            pred_entities = self.decode(sentence, pred)

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def show_stats(self):
        f1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)

            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, f1))
            self.logger.info("Macro-F1: %f" % np.mean(f1_scores))

        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in self.stats_dict])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in self.stats_dict])
        true_entities = sum([self.stats_dict[key]["样本实体数"] for key in self.stats_dict])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_entities + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info(f"Micro-F1: {micro_f1:.4f}")
        self.logger.info("-" * 50)

    @staticmethod
    def decode(sentence, labels):
        # 确保标签是整数列表
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        labels_str = ''.join(str(x) for x in labels)
        results = defaultdict(list)

        patterns = {
            "LOCATION": r"(04+)",
            "ORGANIZATION": r"(15+)",
            "PERSON": r"(26+)",
            "TIME": r"(37+)"
        }

        for entity, pattern in patterns.items():
            for match in re.finditer(pattern, labels_str):
                s, e = match.span()
                if e <= len(sentence):
                    results[entity].append(sentence[s:e])

        return results
