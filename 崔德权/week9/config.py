# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "pretrain_model_path":r"D:\badou\pretrain_model\chinese_bert_likes\bert-base-chinese",
    "max_length": 50,
    "epoch": 10,
    "batch_size": 10,
    "optimizer": "adam",
    "learning_rate":1e-5,
    "seed":42,
    "num_labels": 3,
    "recurrent":"gru",
    "max_sentence": 50
}

if "CUDA_VISIBLE_DEVICES" in os.environ:
    Config["num_gpus"] = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
else:
    Config["num_gpus"] = 0

Config["train_data_path"] = "data/train.txt"
Config["valid_data_path"] = "data/test.txt"

