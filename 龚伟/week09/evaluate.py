import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data


"""
模型效果测试
"""


class Evaluator: #实现了一个深度学习模型评估器的基本框架，主要用于模型验证阶段的流程管理
    def __init__(self, config, model, logger):
        self.config = config #存储验证数据集路径、评估参数等配置信息
        self.model = model #model：待评估的模型实例，需实现forward方法
        self.logger = logger #日志记录器，用于输出评估指标
        self.valid_data = load_data(  #通过load_data加载的验证数据集，设置shuffle=False保证评估顺序固定
            config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch): #该评估方法实现了命名实体识别(NER)任务的模型验证流程
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int), #初始化四类实体（地点/时间/人物/组织）的统计字典，使用defaultdict自动处理未登录实体
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval() #切换模型至评估模式，关闭Dropout等训练专用层
        # print("数据集首条样本结构:", self.valid_data.dataset[0].keys())
        for index, batch_data in enumerate(self.valid_data):

            sentences = ["".join(item['raw_text']) for item in self.valid_data.dataset[
                                                      index * self.config["batch_size"]: (index + 1) * self.config[
                                                          "batch_size"]
                                                      ]]
            print(f"当前batch索引: {index}, 数据类型: {type(batch_data)}")
            print(type(batch_data))  # 检查实际数据结构
            print(batch_data.keys() if isinstance(batch_data, dict) else len(batch_data))

            if isinstance(batch_data, dict):  # 字典类型处理
                input_id = batch_data['input_ids']
                attention_mask = batch_data['attention_mask']
                labels = batch_data['labels']
            else:  # 列表/元组类型处理
                input_id = batch_data[0]
                attention_mask = batch_data[1]
                labels = batch_data[2]
            if torch.cuda.is_available():
                input_id = input_id.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                print(f"当前模型配置: {self.model.config}")
                print(f"return_dict默认值: {self.model.config.return_dict}")
                outputs = self.model(input_ids=input_id, attention_mask=attention_mask,labels=labels)
                logits = outputs[0]
                print(type(outputs))  # 验证输出类型:ml-citation{ref="3" data="citationList"}
                print(dir(outputs))
                print(f"模型返回类型配置: {self.model.config.return_dict}")
                print(f"模型输出结构: {type(outputs)}")
                if isinstance(outputs, tuple):  # 元组类型处理
                    pred_results = outputs[0]  # 第一个元素为logits
                elif hasattr(outputs, 'logits'):  # ModelOutput类型处理
                    pred_results = outputs.logits
                else:  # 兜底处理
                    pred_results = outputs

            self.write_stats(labels, pred_results, sentences) #预测结果与标签通过write_stats方法进行比对统计
        self.show_stats() #最终调用show_stats输出各类实体识别指标（应包含精确率/召回率等）
        return

    def write_stats(self, labels, pred_results, sentences): #实现了命名实体识别任务的评估指标统计功能
        assert len(labels) == len(pred_results) == len(sentences) #通过assert确保标签、预测结果和原始句子的数量一致性
        if not self.config["use_crf"]: #根据config["use_crf"]配置决定是否对预测结果取argmax
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist() #将GPU张量数据转移至CPU并转换为Python列表格式
            true_entities = self.decode(sentence, true_label) #调用decode方法将标签序列转换为结构化实体字典（按实体类型分类存储）
            pred_entities = self.decode(sentence, pred_label)

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]: #对四类实体分别统计三个核心指标
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self): #实现了命名实体识别任务的评估指标计算与输出功能
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]: #遍历四类实体（人物/地点/时间/组织），分别计算
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / \
                (1e-5 + self.stats_dict[key]["识别出实体数"])    #精确率（Precision）：正确识别数/模型预测总数
            recall = self.stats_dict[key]["正确识别"] / \
                (1e-5 + self.stats_dict[key]["样本实体数"])     #召回率（Recall）：正确识别数/真实实体总数
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)   #F1值：精确率与召回率的调和平均数  添加1e-5平滑系数避免除零错误
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" %   #使用logger.info输出每类实体的详细指标
                             (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))  #使用logger.info输出每类实体的详细指标
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in [  #汇总所有类别的统计量（微观指标计算）
                           "PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"]
                         for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"]
                        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / \
            (micro_precision + micro_recall + 1e-5)  #计算Macro-F1：对各类别F1值取算术平均，反映模型在各类别上的整体表现
        self.logger.info("Micro-F1 %f" % micro_f1)

        return

    '''
    {
      "B-LOCATION": 0,  地点
      "B-ORGANIZATION": 1,  组织
      "B-PERSON": 2,   人物
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''

    def decode(self, sentence, labels): #实现了命名实体识别任务中的标签解码功能
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)  #自动初始化实体类型容器
        for location in re.finditer("(04+)", labels): #使用正则表达式匹配连续实体标签 (04+)匹配地点实体（B-LOCATION后接I-LOCATION）
            s, e = location.span()  #通过span()获取实体在句子中的起止位置
            results["LOCATION"].append(sentence[s:e])
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
