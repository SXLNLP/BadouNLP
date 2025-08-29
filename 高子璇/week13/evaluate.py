# -*- coding: utf-8 -*-
import torch
from loader import load_data
from sklearn.metrics import classification_report

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        all_labels = []
        all_preds = []
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)[0]
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(torch.argmax(pred_results, dim=2).cpu().numpy().flatten())
        self.show_stats(all_labels, all_preds)

    def show_stats(self, all_labels, all_preds):
        report = classification_report(all_labels, all_preds, target_names=self.config["ner_tags"], zero_division=0)
        self.logger.info(report)
        self.logger.info("--------------------")

if __name__ == "__main__":
    from config import Config
    from model import TorchModel
    from peft import get_peft_model, LoraConfig

    tuning_tactics = Config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    model = TorchModel
    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load('output/lora_tuning.pth'))
    model = model.cuda()
    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(0)
