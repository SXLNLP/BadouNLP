# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_ner.json",
    "valid_data_path": "data/valid_ner.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "tuning_tactics": "lora_tuning",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"E:\pretrain_models\bert-base-chinese",
    "seed": 987,
    "ner_tags": ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"],  # NER标签
}
