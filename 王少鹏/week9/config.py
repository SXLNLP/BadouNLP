# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "ner_wrok/model_output",
    "schema_path": "ner_wrok/ner_data/schema.json",
    "train_data_path": "ner_wrok/ner_data/train",
    "valid_data_path": "ner_wrok/ner_data/test",
    "vocab_path": "ner_wrok/chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"/home/nbs07/model/bert-base-chinese"
}

