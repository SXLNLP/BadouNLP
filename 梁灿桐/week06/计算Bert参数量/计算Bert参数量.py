import torch
from transformers import BertModel

# 加载预训练模型
bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

# 计算参数量
total_params = 0
for name, param in state_dict.items():
    total_params += param.numel()

print(f"BERT模型的总参数量: {total_params}")
print(f"详细参数分布:")
print("-" * 50)
for name, param in state_dict.items():
    print(f"{name:<60} | 形状: {str(param.shape):<20} | 参数量: {param.numel():>10,}")
print("-" * 50)
print(f"参数总量: {total_params:,} (约 {total_params/1e6:.1f}M)")
