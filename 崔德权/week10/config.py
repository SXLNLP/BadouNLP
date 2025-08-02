# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "vocab_path": "vocab.txt",  # 词表文件路径
    "bert_model_path": "bert-base-chinese",  # huggingface模型名或本地目录
    "train_data_path": "data.jsonl",  # 训练数据路径
    "batch_size": 8,
    "input_max_length": 128,
    "output_max_length": 32,
    "lr": 1e-5,
    "epochs": 3
}

