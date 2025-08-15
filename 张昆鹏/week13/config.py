# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"N:\八斗\上一期\第十三周 大模型相关内容第三讲\week13 大语言模型相关第三讲\训练大模型\homework\model_output",
    "schema_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\schema.json",
    "train_data_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\train",
    "valid_data_path": r"N:\八斗\上一期\第九周 序列标注\课件\week9 序列标注问题\homework\ner_data\test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"N:\八斗\上一期\bert-base-chinese"
}

