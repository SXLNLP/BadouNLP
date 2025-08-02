# -*- coding: utf-8 -*-

"""
BERT+Mask自回归语言模型训练配置
"""

Config = {
    # 通用配置
    "seed": 42,  # 随机种子，用于结果复现
    "do_lower_case": True,  # 是否将文本转换为小写
    "use_cuda": True,  # 是否使用GPU
    
    # 数据路径配置
    "train_data_path": "./data/train.txt",  # 训练数据路径
    "valid_data_path": "./data/valid.txt",  # 验证数据路径
    "test_data_path": "./data/test.txt",    # 测试数据路径
    
    # 模型保存配置
    "model_path": "./saved_models",  # 模型保存根目录
    "pretrained_model_path": None,   # 预训练模型路径，如使用预训练模型则指定路径
    "save_steps": 1,                 # 每隔多少个epoch保存一次模型
    
    # 训练参数配置
    "epoch": 10,                     # 训练轮数
    "batch_size": 32,                # 批次大小
    "optimizer": "adam",             # 优化器类型：adam或sgd
    "learning_rate": 1e-4,           # 学习率
    "warmup_proportion": 0.1,        # 学习率warmup比例
    "weight_decay": 0.01,            # 权重衰减率
    "max_grad_norm": 1.0,            # 梯度裁剪阈值
    
    # BERT模型参数配置
    "vocab_size": 21128,             # 词汇表大小，中文BERT默认21128
    "hidden_size": 768,              # BERT隐藏层大小
    "num_hidden_layers": 12,         # BERT层数
    "num_attention_heads": 12,       # 注意力头数量
    "intermediate_size": 3072,       # 中间层大小
    "hidden_act": "gelu",            # 隐藏层激活函数
    "hidden_dropout_prob": 0.1,      # 隐藏层dropout率
    "attention_probs_dropout_prob": 0.1,  # 注意力dropout率
    "max_position_embeddings": 512,  # 最大位置编码
    "type_vocab_size": 2,            # 类型词汇表大小
    "initializer_range": 0.02,       # 初始化范围
    
    # Mask机制参数
    "mlm_probability": 0.15,         # 进行mask的概率
    "max_seq_length": 128,           # 最大序列长度
    
    # 评估配置
    "eval_steps": 500,               # 每隔多少步进行一次评估
    "logging_steps": 100,            # 每隔多少步记录一次日志
}    