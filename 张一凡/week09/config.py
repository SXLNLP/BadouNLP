# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import torch

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "max_length": 128,
    "hidden_size": 768,
    "num_layers": 12,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 5e-5,
    "max_grad_norm": 1.0,
    "use_crf": True,
    "class_num": 9,
    "bert_path": "bert-base-chinese",  # 修改后的配置
    "dropout": 0.1,
    "weight_decay": 0.01,
    "require_improvement": 1000,
    "use_crf": True,  # 或 False
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
