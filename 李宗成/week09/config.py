# -*- coding: utf-8 -*-

"""
配置参数信息
"""

import os

current_path = os.path.dirname(os.path.abspath(__file__))

Config = {
    "model_path": "model_output",
    "schema_path": current_path + "/ner_data/schema.json",
    "train_data_path": current_path + "/ner_data/train",
    "valid_data_path": current_path + "/ner_data/test",
    "vocab_path": current_path + "/chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "use_crf": True,
    "class_num": 9,
    "bert_model": r"D:\\workspace\\pretrain_models\\bert-base-chinese"
}

