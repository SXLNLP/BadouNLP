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
    "max_length": 100,
    "hidden_size": 768,  # BERT的隐藏层大小是768
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT通常使用较小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\bert-base-chinese",
    "bert_vocab_path": r"D:\bert-base-chinese\vocab.txt"  # 添加BERT词表路径
}
