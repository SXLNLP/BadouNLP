# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_data.csv",
    "verify_data_path": "verify_data.csv",
    "bert_model_path":r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese",
    "bert_vocab_path":r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese\vocab.txt",
    "vocab_size": 349046,
    "class_num": 1,
    "jieba_vocab_path": "dict.txt",
    "model_type":"bert",
    "max_length": 512,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 3,
    "epoch": 10,
    "batch_size": 10,
    "use_pooling": True,
    "pooling_style":"AVG",
    "optimizer_type": "Adam",
    "learning_rate": 1e-3,
    "seed": 987,
    "bert_output_hidden_states": False,
    "loss_type": "MSE",
    "activation_type": "sigmoid",
    "data_file_path":"文本分类练习.csv"
}

