# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/data/schema.json",
    "train_data_path": "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/data/train.json",
    "valid_data_path": "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/data/valid_fixed.json",
    "vocab_path": "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/chars.txt",
    "vocab_size": 4622,
    "max_length": 20,
    "hidden_size": 128,
    "embed_size": 100,
    "epoch": 10,
    "batch_size": 128,
    "epoch_data_size": 2000,     # 每轮训练中采样数量（可用于 triplet 的 anchor 数量）
    "num_neg_samples": 1,       # 每个 anchor 对应的负样本数量（可选，默认1）
    "triplet_margin": 1.0,      # Triplet Loss 中的 margin 参数
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
