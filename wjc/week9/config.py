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
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 30,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT需要更小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\BaiduNetdiskDownload\badouai\recored\bert-base-chinese",
    "use_bert": False,  # 新增标志
    "bert_hidden_size": 768,  # 新增
    "freeze_bert": True  # 新增参数：冻结BERT底层
}
