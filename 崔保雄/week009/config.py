# -*- coding: utf-8 -*-

"""
    配置参数信息
    在原先基础上，增加了一个配置model_type，代表使用lstm还是bert
"""

Config = {
    "model_path": "model_output",
    "model_type": "bert", #lstm bert
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\www.root\bert-base-chinese"
}
