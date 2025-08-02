# -*- coding: utf-8 -*-
import torch
import os
from config import Config
from loader import load_vocab
from model import TorchModel

"""
模型预测工具类
"""


class Predictor:
    def __init__(self, config, model_path):
        """
        初始化预测器
        :param config: 配置参数（来自config.py的Config）
        :param model_path: 训练好的模型路径（如"model_output/epoch_20.pth"）
        """
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])  # 加载字表/词表
        self.schema = self.load_schema(config["schema_path"])  # 加载标签映射（如"B-PERSON":0）
        self.label_list = self.reverse_schema(self.schema)  # 反向映射（0:"B-PERSON"）
        self.model = self.load_model(model_path)  # 加载模型
        self.cuda_flag = torch.cuda.is_available()  # 是否使用GPU
        if self.cuda_flag:
            self.model = self.model.cuda()

    def load_schema(self, path):
        """加载标签到ID的映射（同loader中的逻辑）"""
        import json
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def reverse_schema(self, schema):
        """将{标签:ID}转换为{ID:标签}，用于最终结果映射"""
        return {v: k for k, v in schema.items()}

    def load_model(self, model_path):
        """加载训练好的模型参数"""
        model = TorchModel(self.config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 支持CPU加载
        model.eval()  # 切换到评估模式
        return model

    def preprocess(self, text):
        """预处理输入文本：转为ID并补齐长度"""
        input_id = []
        # 按字分割（如果是词表则用jieba，但这里默认用字表）
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))  # 未知字符用[UNK]的ID
        # 补齐到max_length
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))  # 0是padding的ID
        return torch.LongTensor([input_id])  # 增加batch维度

    def decode(self, text, labels):
        """将模型输出的标签ID转换为实体结果（参考evaluate.py的decode逻辑）"""
        entities = {}
        # 过滤padding部分，只保留文本长度内的标签
        valid_labels = labels[:len(text)]
        # 转换为标签字符串（如0→"B-LOCATION"）
        label_str = [self.label_list[label_id] for label_id in valid_labels]

        # 提取实体（以PERSON为例，其他实体类似）
        entities["PERSON"] = self.extract_entity(text, label_str, "PERSON")
        entities["LOCATION"] = self.extract_entity(text, label_str, "LOCATION")
        entities["TIME"] = self.extract_entity(text, label_str, "TIME")
        entities["ORGANIZATION"] = self.extract_entity(text, label_str, "ORGANIZATION")
        return entities

    def extract_entity(self, text, labels, entity_type):
        """提取指定类型的实体（如PERSON）"""
        entities = []
        current_entity = ""
        for char, label in zip(text, labels):
            if label == f"B-{entity_type}":
                # 开始一个新实体
                if current_entity:  # 如果之前有未结束的实体，先加入
                    entities.append(current_entity)
                current_entity = char
            elif label == f"I-{entity_type}":
                # 实体延续
                current_entity += char
            else:
                # 实体结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        return entities

    def predict(self, text):
        """完整预测流程：预处理→模型预测→解码实体"""
        # 预处理
        input_tensor = self.preprocess(text)
        if self.cuda_flag:
            input_tensor = input_tensor.cuda()
        # 模型预测（不传入target，返回预测结果）
        with torch.no_grad():  # 关闭梯度计算
            pred_labels = self.model(input_tensor)  # 输出是标签ID（CRF返回解码结果，非CRF返回argmax结果）
        # 如果使用CRF，输出是list[list]（batch内的每个样本）；否则是tensor，需要转换
        if not self.config["use_crf"]:
            pred_labels = torch.argmax(pred_labels, dim=-1).cpu().numpy()[0]  # 取第一个样本
        else:
            pred_labels = pred_labels[0]  # CRF返回的是list，直接取第一个样本
        # 解码为实体
        entities = self.decode(text, pred_labels)
        return {
            "text": text,
            "entities": entities
        }


