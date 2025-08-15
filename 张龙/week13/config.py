# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "../model_output",
    "schema_path": "../ner_data/schema.json",
    "train_data_path": "../ner_data/train",
    "valid_data_path": "../ner_data/test",
    "vocab_path": "../chars.txt",
    "max_length": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "epoch": 10,
    "tuning_tactics": "lora_tuning",
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "use_bert": True,
    "pretrain_model_path": r"D:\nlp\第六周 语言模型\bert-base-chinese\bert-base-chinese"
}

