# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"./bert-base-chinese/vocab.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,  # Reduced learning rate for BERT
    "use_crf": True,
    "class_num": 9,
    "bert_path": "bert-base-chinese",
    "bert_num_layers": 6  # Reduced from 12 for small dataset
}

