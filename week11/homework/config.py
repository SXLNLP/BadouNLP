# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "input_max_length": 30,
    "output_max_length": 120,
    "epoch": 600,
    "batch_size": 104,
    "optimizer": "adam",
    "learning_rate":15e-4,
    "seed":42,
    "vocab_size":6219,
    "vocab_path":"vocab.txt",
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "beam_size":5
    }

