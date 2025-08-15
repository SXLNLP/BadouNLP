# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, labels in self.valid_data:
                if torch.cuda.is_available():
                    input_ids, labels = input_ids.cuda(), labels.cuda()
                outputs = self.model(input_ids)
                predictions = torch.argmax(outputs.logits, dim=2)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        accuracy = correct / total
        self.logger.info(f"准确率: {accuracy:.4f}")
        return accuracy
