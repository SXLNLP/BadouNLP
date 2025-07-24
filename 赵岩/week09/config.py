# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 128,
    "hidden_size": 768,
    "num_layers": 3,
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"E:\BaiduNetdiskDownload\bert-base-chinese"
}

