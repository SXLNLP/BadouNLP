from transformers import BertConfig, BertForMaskedLM
import torch

# 从 config.json 文件加载配置
config = BertConfig.from_json_file("config.json")

# 根据配置创建 BERT 模型（MaskedLM 结构）
model = BertForMaskedLM(config)

# 计算参数总量
total_params = sum(p.numel() for p in model.parameters())

# 可选：查看可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
"""
输出：
总参数量: 102,290,312
可训练参数量: 102,290,312
"""
