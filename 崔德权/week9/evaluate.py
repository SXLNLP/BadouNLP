# -*- coding: utf-8 -*-
import torch
import collections
import io
import json
import six
import sys
import argparse
from loader import load_data
from collections import defaultdict, OrderedDict
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, logger, shuffle=False)
        self.tokenizer = self.valid_data.tokenizer

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = defaultdict(int)  # 用于存储测试结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, attention_mask, label = batch_data
            with torch.no_grad():
                pred_label = self.model(input_ids, attention_mask)
                pred_label = torch.argmax(pred_label, -1)
            self.write_stats(pred_label, label)
        self.show_stats()
        return
    
    def write_stats(self, pred_label, label):
        assert len(pred_label) == len(label), (pred_label.shape, label.shape)
        label = label.squeeze()
        pred_label = [int(x) for x in pred_label]
        true_label = [int(x) for x in label]
        self.chuck_acc_stats(pred_label, true_label)
        return


    def chuck_acc_stats(self, pred_label, true_label):
        gold_spans = set(self.get_chucks(true_label))
        pred_spans = set(self.get_chucks(pred_label))

        self.stats_dict["total_gold"] += len(gold_spans)
        self.stats_dict["total_pred"] += len(pred_spans)
        self.stats_dict["p"] += len(pred_spans.intersection(gold_spans))
        return 
    
    #@staticmethod
    def get_chucks(self, true_label):
        chucks = []
        start, end = -1, -1
        for index, label in enumerate(true_label):
            #print(index, label)
            if label == 0:
                if start == -1:
                    start = index
                    end = start + 1
                else:
                    chucks.append((start, end))
                    start = index
                    end = start + 1
            elif label == 1:
                end += 1
            elif label == 2:
                if end != -1:
                    chucks.append((start, end))
                    start, end = -1, -1
            else:
                assert False
        if end != -1:
            chucks.append((start, end))
        return chucks

    def show_stats(self):
        total_pred = self.stats_dict["total_pred"]
        total_gold = self.stats_dict["total_gold"]
        p = self.stats_dict["p"]
        precision_e2e = p * 1.0 / total_pred * 100
        recall_e2e = p * 1.0 / total_gold * 100
        fscore_e2e = 2.0 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e)
        self.logger.info("Precision: %f\tRecall: %f\tF1 score: %f"%(precision_e2e, recall_e2e, fscore_e2e))
        self.logger.info("--------------------")
        return




if __name__ == "__main__":
    label = [2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    print([(i, l) for i, l in enumerate(label)])
    print(Evaluator.get_chucks(label))
