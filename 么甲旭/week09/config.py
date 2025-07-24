# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "max_length": 100,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,  # BERT通常使用较小的学习率
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\BaiduNetdiskDownload\AI架构课程\第六周 语言模型\week6\bert-base-chinese"
}
