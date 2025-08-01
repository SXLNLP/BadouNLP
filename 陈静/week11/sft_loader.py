# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
SFT训练数据加载器
将新闻数据转换为SFT格式：指令+输入+输出
"""

class SFTDataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        
        # SFT指令模板
        self.instruction_templates = [
            "请为以下新闻内容生成合适的标题：",
            "根据以下新闻内容，生成一个准确的新闻标题：",
            "请阅读以下新闻并提供简洁的标题：",
            "为这篇新闻写一个恰当的标题：",
            "请总结以下新闻内容并生成标题："
        ]
        
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_sft_data(title, content, i)
        return

    def encode_sentence(self, text, max_length, with_cls_token=True, with_sep_token=True):
        """文本编码为token ids"""
        input_id = []
        if with_cls_token:
            input_id.append(self.vocab["[CLS]"])
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if with_sep_token:
            input_id.append(self.vocab["[SEP]"])
        input_id = self.padding(input_id, max_length)
        return input_id

    def padding(self, input_id, length):
        """补齐或截断序列"""
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    def prepare_sft_data(self, title, content, index):
        """
        准备SFT格式的训练数据
        格式：[指令] + [新闻内容] + [目标标题]
        """
        # 选择指令模板
        instruction = self.instruction_templates[index % len(self.instruction_templates)]
        
        # 构建完整的输入序列：指令 + 内容
        full_input = instruction + content
        
        # 编码输入序列（指令+内容）
        input_seq = self.encode_sentence(full_input, self.config["input_max_length"], False, False)
        
        # 编码目标序列（标题）- 用于decoder输入
        target_seq = self.encode_sentence(title, self.config["output_max_length"], True, False)
        
        # 编码黄金标准（标题）- 用于计算损失
        gold_seq = self.encode_sentence(title, self.config["output_max_length"], False, True)
        
        # 为SFT训练创建attention mask，只对response部分计算损失
        # 这里我们简化处理，如果需要更精确的控制，可以添加更复杂的mask逻辑
        
        self.data.append([
            torch.LongTensor(input_seq),    # 编码器输入（指令+内容）
            torch.LongTensor(target_seq),   # 解码器输入（标题，用于teacher forcing）
            torch.LongTensor(gold_seq),     # 标签（标题，用于计算损失）
        ])
        
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    """加载词汇表"""
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict


def load_sft_data(data_path, config, logger, shuffle=True):
    """加载SFT训练数据"""
    dg = SFTDataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


# 创建SFT训练配置
def create_sft_config(base_config):
    """基于原配置创建SFT训练配置"""
    sft_config = base_config.copy()
    
    # SFT特有配置
    sft_config.update({
        "sft_mode": True,                    # 启用SFT模式
        "instruction_loss_weight": 0.0,      # 指令部分损失权重（通常为0）
        "response_loss_weight": 1.0,         # 回答部分损失权重
        "learning_rate": 5e-5,               # SFT通常使用较小的学习率
        "epoch": 50,                         # SFT通常需要较少的epoch
        "warmup_steps": 100,                 # 预热步数
    })
    
    return sft_config


if __name__ == "__main__":
    from config import Config
    
    # 测试SFT数据加载
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    sft_config = create_sft_config(Config)
    dl = load_sft_data(sft_config["train_data_path"], sft_config, logger)
    
    print("SFT数据加载测试：")
    for batch_data in dl:
        input_seq, target_seq, gold_seq = batch_data
        print(f"输入序列形状: {input_seq.shape}")
        print(f"目标序列形状: {target_seq.shape}")
        print(f"标签序列形状: {gold_seq.shape}")
        break