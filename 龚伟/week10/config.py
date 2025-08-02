import os
import torch

Config = {
    # 基础路径
    "model_path": "output",
    "bert_path": "bert-base-chinese",  # Bert模型路径
    "vocab_path": "vocab.txt",
    # 数据参数
    "input_max_length": 256,  # 新闻内容最长长度（Bert适合更长输入）
    "output_max_length": 30,   # 标题最长长度
    "train_data_path": "sample_data.json",
    "valid_data_path": "sample_data.json",
    # 训练参数
    "epoch": 50,  # Bert微调无需太多epoch
    "batch_size": 8,  # Bert显存需求高，减小batch
    "learning_rate": 2e-5,  # Bert微调推荐学习率（远小于原1e-3）
    "seed": 42,
    # 生成解码参数
    "beam_size": 3,
    "n_gram_size": 2,  # 用于N-gram惩罚（避免2-gram重复）
    "repetition_penalty": 1.2,  # 重复惩罚系数（>1抑制重复）
    # 模型参数（Bert固定，无需手动定义）
    "vocab_size":6219,
    "pad_idx": 0,  # Bert默认[PAD]索引为0
    "start_idx": 101,  # Bert默认[CLS]索引为101（作为生成起始）
    "end_idx": 102,    # Bert默认[SEP]索引为102（作为生成结束）
}