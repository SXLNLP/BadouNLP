# -*- coding: utf-8 -*-
# 导入必要的库
import torch  # PyTorch深度学习框架
import re     # 正则表达式模块，用于模式匹配
import numpy as np  # 数值计算库
from collections import defaultdict  # 当访问不存在的键时返回默认值的字典
from loader import load_data  # 自定义数据加载函数

"""
模型效果测试模块
这个模块包含用于评估命名实体识别(NER)模型性能的类和方法
"""

class Evaluator:
    """
    Evaluator类用于评估NER模型在验证集上的性能
    
    参数:
        config: 配置字典，包含模型和数据的各种配置参数
        model: 训练好的NER模型
        logger: 日志记录器，用于输出评估结果
    """
    
    def __init__(self, config, model, logger):
        """初始化Evaluator对象"""
        self.config = config  # 保存配置信息
        self.model = model    # 保存模型
        self.logger = logger  # 保存日志记录器
        # 加载验证数据
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # config["valid_data_path"]: 验证数据文件路径
        # config: 数据配置
        # shuffle=False: 不打乱验证数据顺序

    def eval(self, epoch):
        """
        评估模型在给定epoch的性能
        
        参数:
            epoch: 当前训练轮次，用于日志记录
        """
        # 记录开始测试的日志
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        
        # 初始化统计字典，用于存储不同实体类型的评估指标
        # 每个实体类型对应一个defaultdict(int)，记录各种计数
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        
        self.model.eval()  # 将模型设置为评估模式
        
        # 遍历验证数据
        for index, batch_data in enumerate(self.valid_data):
            # 获取当前batch的原始句子
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: 
                      (index+1) * self.config["batch_size"]]
            
            # 如果有GPU可用，将数据移动到GPU
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]  # 将每个数据项移到GPU
            
            input_id, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            
            with torch.no_grad():  # 禁用梯度计算
                # 使用模型进行预测（不输入labels）
                pred_results = self.model(input_id)
            
            # 更新评估指标统计
            self.write_stats(labels, pred_results, sentences)
        
        self.show_stats()  # 显示最终评估结果
        return

    def write_stats(self, labels, pred_results, sentences):
        """
        更新评估统计信息
        
        参数:
            labels: 真实标签
            pred_results: 模型预测结果
            sentences: 原始文本句子
        """
        assert len(labels) == len(pred_results) == len(sentences)  # 确保三个列表长度一致
        
        # 如果没有使用CRF层，则取最大概率作为预测结果
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        
        # 对每个样本进行处理
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            # 如果没有使用CRF，将预测结果转为CPU上的列表
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            
            # 将真实标签转为CPU上的列表
            true_label = true_label.cpu().detach().tolist()
            
            # 解码得到真实实体和预测实体
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            
            # 更新统计信息
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                # 正确识别的实体数：预测和真实都存在的实体
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                # 样本实体数：真实存在的实体数量
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                # 识别出实体数：模型预测的实体数量
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        """显示评估统计信息，包括准确率、召回率和F1分数"""
        F1_scores = []  # 存储各个类别的F1分数
        
        # 计算并记录每个实体类型的评估指标
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 计算准确率：正确识别的实体数 / 识别出的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            # 计算召回率：正确识别的实体数 / 样本中的实体数
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            # 计算F1分数
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)  # 保存F1分数
            
            # 记录评估结果
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        
        # 计算宏平均F1分数
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        
        # 计算微平均指标
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        
        # 计算微平均准确率、召回率和F1分数
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        
        # 记录微平均F1分数
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
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
        """
        将模型输出的标签解码为具体的实体
        
        参数:
            sentence: 原始文本句子
            labels: 模型预测或真实的标签序列
            
        返回:
            包含各类实体的字典，每个实体类型对应一个实体列表
        """
        # 将标签序列转换为字符串形式
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        
        results = defaultdict(list)  # 存储解码结果
        
        # 使用正则表达式查找LOCATION实体(B-LOCATION后跟多个I-LOCATION)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()  # 获取匹配的位置范围
            results["LOCATION"].append(sentence[s:e])  # 提取对应的文本片段
        
        # 类似地查找ORGANIZATION实体(B-ORG后跟多个I-ORG)
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        
        # 查找PERSON实体(B-PER后跟多个I-PER)
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        
        # 查找TIME实体(B-TIME后跟多个I-TIME)
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        
        return results
