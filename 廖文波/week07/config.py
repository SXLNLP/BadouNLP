# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/train_data.json",
    "valid_data_path": "week7 文本分类问题/week7 文本分类问题/work_nn_pipline/basedata/valid_data.json",
    "vocab_path":"week7 文本分类问题/week7 文本分类问题/nn_pipline/chars.txt",
    "model_type":"bert",  # 可选值: lstm, cnn, bert, bert_lstm, bert_cnn, bert_mid_layer
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/home/nbs07/model/bert-base-chinese",
    "seed": 987
}

