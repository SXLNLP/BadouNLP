# -*- coding: utf-8 -*-
import torch
from loader import load_data
import pandas as pd
import openpyxl

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.sentences = self.valid_data.dataset.sentences  # ==> 获取句子列表
        self.stats_dict = {"correct": 0, "wrong": 0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        # ==> 记录评估结果
        self.writer = openpyxl.Workbook()
        self.sheet = self.writer.active
        self.sheet.append(["sentence", "true_label", "pred_label", "is_correct"])
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids)  #不输入labels，使用模型当前参数进行预测
            # self.write_stats(labels, pred_results)
            self.write_stats_2(labels, pred_results, self.sentences[index*self.config["batch_size"]:(index+1)*self.config["batch_size"]])
        acc = self.show_stats()
        self.writer.save("/Users/juju/nlp20/class7/hwAndPrac/hw/output/valid_result.xlsx")
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    """
    写状态和sheet的结果
    """
    def write_stats_2(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            pred_label = torch.argmax(pred_label)
            self.sheet.append([sentence, int(true_label), int(pred_label), int(true_label) == int(pred_label)])
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
