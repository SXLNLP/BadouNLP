# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "廖文波/week08/model_output",
    "schema_path": "廖文波/week08/data/schema.json",
    "train_data_path": "廖文波/week08/data/train.json",
    "valid_data_path": "廖文波/week08/data/valid.json",
    "vocab_path":"廖文波/week08/chars.txt",
    "model_type":"bert",
    "pretrain_model_path":r"/home/nbs07/model/bert-base-chinese",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-5,
}