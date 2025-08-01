# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "data_path": "sample_data.json",
    "test_data_path": "test.txt",
    "vocab_path":"vocab.txt",
    "max_length": 140,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "save_interval": 1,
    "batch_size": 20,
    "optimizer": "adam",
    "learning_rate": 3e-5,
    "use_crf": True,
    "train_sample": 50000,
    "bert_path": r"D:\PycharmProjects\bert-base-chinese",
    "repetition_penalty_start": 1.0,  # 初始重复惩罚强度
    "repetition_penalty_end" : 1.5,    # 最终重复惩罚强度
    "negative_sample_ratio" : 0.1  ,    # 负样本比例
    "generation_temperature" : 0.85 ,   # 生成温度
    "generation_top_k" : 30         # 生成时top-k值
}

