# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./model_output",
    "schema_path": "./ner_data/schema.json",
    "train_data_path": "./ner_data/train",
    "valid_data_path": "./ner_data/test",
    "vocab_path":"../../bert-base-chinese/vocab.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 3,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": True,
    "class_num": 9,
    "pretrain_model_path":r"../../bert-base-chinese",
}

