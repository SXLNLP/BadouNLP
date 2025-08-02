# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "input_max_length": 120,  # 输入(content)最大长度
    "output_max_length": 30,  # 输出(title)最大长度
    "epoch": 20,             # 训练轮数
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-4,   # 学习率
    "seed": 42,
    "vocab_size": 6219,
    "vocab_path": "vocab.txt",
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "beam_size": 5,          # beam search宽度
    "pad_idx": 0,            # PAD标记的索引
    "start_idx": 2,          # CLS标记的索引
    "end_idx": 3             # SEP标记的索引
}
