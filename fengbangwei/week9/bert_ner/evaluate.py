# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
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
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # 按batch_size 获取验证数据
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attention_mask, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id, attention_mask)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        # 如果不满足条件，说明存在严重的数据对齐问题，应立即终止评估流程
        # 当三个输入的长度不一致时抛出异常，表明存在严重的数据对齐问题
        assert len(labels) == len(pred_results) == len(sentences)
        # 原始输出 (使用CRF时)
        # pred_results: [["B-PER", "I-PER", "O", ...], ...]
        # 原始输出 (未使用CRF时)
        # pred_results: torch.Tensor([[0.1,0.2,...], [...]])
        # → 需要argmax和转换
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            # .cpu()：将GPU上的张量转移到CPU
            # .detach()：切断计算图，防止误反向传播
            # .tolist()：将张量转为Python列表
            # 仅在不使用CRF时需要，因为CRF输出已是解码后的序列
            if not self.config["use_crf"]:
                pred_label1 = pred_label.cpu().detach().tolist()
            true_label1 = true_label.cpu().detach().tolist()

            true_label = []
            pred_label = []
            for t, p in zip(true_label1, pred_label1):
                if t != -1:
                    true_label.append(t)
                    pred_label.append(p)
            sentence = sentence[:len(true_label)]
            pred_label = pred_label[:len(sentence)]
            true_label = true_label[:len(sentence)]
            true_entities = self.decode(sentence, pred_label)
            pred_entities = self.decode(sentence, true_label)
            # print("=+++++++++")
            # print(true_entities)
            # print(pred_entities)
            # print('=+++++++++')
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                # 假设评估LOCATION类别：
                # pred_locations = ["北京", "上海", "广州"]  # 模型预测的地点
                # true_locations = ["北京", "广州"]  # 真实标注的地点
                # correct = [loc for loc in pred_locations if loc in true_locations]
                # 结果：["北京", "广州"]（上海未被包含）
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
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
        self.logger.info("宏观Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("微观Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    解码实体
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
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        # 使用正则表达式在标签序列中查找所有连续出现的"04"模式
        # 0：对应B - LOCATION标签（起始标签）
        # 4 +：对应I - LOCATION标签（延续标签），+表示1次或多次重复
        # ()：捕获分组，用于获取完整匹配范围 假设标签序列为："004440000"  匹配结果："0444"（B-LOCATION + I-LOCATION*3）
        for location in re.finditer("(04+)", labels):
            s, e = location.span()  # 例如匹配到(1,5)
            results["LOCATION"].append(sentence[s:e])  # 提取对应位置的文本作为实体
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
