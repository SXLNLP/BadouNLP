# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese\vocab.txt",
    "max_length": 512,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 1,
    "batch_size": 5,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese"
}

