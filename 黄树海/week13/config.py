# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 10,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 64,
    #"tuning_tactics":"lora_tuning",
    "tuning_tactics":"prefix_tuning",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\nlp516\bert-base-chinese",
    "seed": 987
}