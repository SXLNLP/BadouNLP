# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import classification_report
from loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.reset_stats()

    def reset_stats(self):
        self.stats_dict = {
            "correct": 0,
            "wrong": 0,
            "total": 0
        }
        self.y_true = []
        self.y_pred = []

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.model.eval()
        self.reset_stats()

        with torch.no_grad():
            for batch in self.valid_data:
                inputs, labels = batch
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self.model(inputs)
                self.update_stats(outputs, labels)

        self.show_stats()
        return self.stats_dict["correct"] / self.stats_dict["total"]

    def update_stats(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        labels = labels.squeeze()

        correct = (preds == labels).sum().item()
        total = labels.size(0)

        self.stats_dict["correct"] += correct
        self.stats_dict["wrong"] += total - correct
        self.stats_dict["total"] += total

        self.y_true.extend(labels.cpu().numpy())
        self.y_pred.extend(preds.cpu().numpy())

    def show_stats(self):
        acc = self.stats_dict["correct"] / self.stats_dict["total"]
        self.logger.info(f"准确率: {acc:.4f}")

        if self.config["class_num"] == 2:
            report = classification_report(
                self.y_true, self.y_pred,
                target_names=list(self.config["label_map"].keys()),
                digits=4
            )
            self.logger.info("\n分类报告:\n" + report)
