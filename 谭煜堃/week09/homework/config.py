# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "./homework/ner_data/schema.json",
    "train_data_path": "./homework/ner_data/train",
    "valid_data_path": "./homework/ner_data/test",
    "vocab_path":"./homework/chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"E:\pretrain_models\bert-base-chinese",
    "vocab_size": 4622
}

