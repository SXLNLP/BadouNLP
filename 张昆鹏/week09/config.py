# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\model_output",
    "schema_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\schema.json",
    "train_data_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\train",
    "valid_data_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\bert-base-chinese"
}

