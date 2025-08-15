# -*- coding: utf-8 -*-

"""
BERT + LoRA NER序列标注任务配置参数
"""

Config = {
    "model_path": "output",
    "train_data_path": "/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/train",
    "valid_data_path": "/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/test", 
    "schema_path": "/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/schema.json",
    "vocab_path": "/Users/evan/Downloads/AINLP/week13 大语言模型相关第三讲/训练大模型/理解lora/chars.txt",
    "model_type": "bert",  # 使用BERT模型进行NER任务
    "max_length": 128,  # NER任务通常需要较长的序列长度
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "class_num": 9,  # BIO格式的NER标签数量
    "epoch": 5,  # NER任务通常需要较少的epoch
    "batch_size": 16,  # 适中的批次大小
    "tuning_tactics": "lora_tuning",  # 使用LoRA微调
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-5,  # NER任务建议使用较小的学习率
    "pretrain_model_path": "/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese",
    "seed": 987
}