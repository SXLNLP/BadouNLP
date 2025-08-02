# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "vocab_size": 20000,
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 3,
    "epoch": 15,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": True,
    "class_num": 9,
    # "bert_path": r"F:\Desktop\work_space\pretrain_models\bert-base-chinese"
    "bert_path": "bert-base-chinese",
    "use_bert": True
}