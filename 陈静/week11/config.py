# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "input_max_length": 120,
    "output_max_length": 30,
    "epoch": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate":1e-3,
    "seed":42,
    "vocab_size":6219,
    "vocab_path":"vocab.txt",
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "beam_size":5
    }

