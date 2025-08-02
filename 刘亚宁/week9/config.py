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
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 3e-5,
    "use_crf": True,
    "class_num": 9,
    "use_bert":True,
    "bert_path": r"E:\BADOU\第六周 语言模型\bert-base-chinese\bert-base-chinese"
}
