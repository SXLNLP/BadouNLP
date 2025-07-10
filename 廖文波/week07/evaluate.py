
import torch
from data_loader import load_data
import numpy as np


class Evaluator:
    def __init__(self, config, logger):
        self.configs = config
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}

    def eval(self, model, epoch):
        self.logger.info("开始第%d轮训练" % epoch)
        # model.eval()  # 确保进入评估模式
        self.stats_dict = {"correct":0, "wrong":0} ##初始化统计参数
        for index, data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                batch_data = [d.to(device) for d in data]
            input_ids, labels = batch_data
            # print("输入内容", input_ids[0], input_ids.shape)
            with torch.no_grad():
                # print("输入", input_ids, input_ids.shape)
                pred_results = model(input_ids,eval=True)
            # print("预测结果", pred_results[0],pred_results.shape)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc
    
    def write_stats(self, labels, pred_results):
        for label, pred_result in zip(labels, pred_results):
            pred_label = torch.argmax(pred_result)
            # print("标签", int(label), "预测结果", int(pred_label))
            if int(label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return
    
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        acc = correct / total
        self.logger.info("correct: %d, wrong: %d, total: %d, acc: %.2f" % (correct, wrong, total, acc))
        return acc