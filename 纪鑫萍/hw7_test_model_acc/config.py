# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "origin_data_path": "/Users/juju/nlp20/class7/hwAndPrac/hw/data/文本分类练习.csv",
    "train_data_path": "/Users/juju/nlp20/class7/hwAndPrac/hw/data/train_data.csv",
    "valid_data_path": "/Users/juju/nlp20/class7/hwAndPrac/hw/data/valid_data.csv",
    "vocab_path": "chars.txt",  # 词表路径
    "model_type": "bert",  # 选择模型类型
    "max_length": 30,  # 输入序列最大长度
    "hidden_size": 256,  # 隐含层维度
    "kernel_size": 3,  # 卷积核大小
    "num_layers": 2,  # bert内transformer层数
    "epoch": 5,  # 训练轮数
    "batch_size": 128,  # 训练batch大小
    "pooling_style": "max",  # 选择池化方式
    "optimizer": "adam",  # 选择优化器
    "learning_rate": 1e-4,  # 学习率
    "pretrain_model_path": r"/Users/juju/BaDou/bert-base-chinese",
    "seed": 987  # ❓
}
