# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch


Config = {
    "model_path": "model_output",
    "train_data_path": "sample_data.json",
    "valid_data_path": "sample_data.json",
    "vocab_path":"vocab.txt",
    "vocab_size": 21128,  # 字表大小
    "hidden_size": 768,
    "num_layers": 2,
    "epoch_num": 20,      #训练轮数
    "char_dim":768,       #每个字的维度
    "window_size":10,     #样本文本长度
    "batch_size": 128,    #每次训练样本的个数
    "train_sample":10000,  #每轮训练总共训练的样本总数
    "optimizer": "adam",
    "learning_rate": 1e-4,#学习率
    "pretrain_model_path": r"D:/Code/py/八斗nlp/20250622/week6 语言模型和预训练/bert-base-chinese",
    "corpus_path": "corpus.txt",  # 语料库路径
}


