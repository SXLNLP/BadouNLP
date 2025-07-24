from transformers import BertModel
import torch

def print_num_params(model):
    total = 0
    print(f"{'Module':60} {'Param #':>12}")
    print("="*75)
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        print(f"{name:60} {num:12,}")
    print("="*75)
    print(f"{'Total':60} {total:12,}")

def print_grouped_params(model):
    groups = {
        'embeddings': 0,
        'encoder': 0,
        'pooler': 0,
        'others': 0
    }
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            groups['embeddings'] += param.numel()
        elif name.startswith('encoder'):
            groups['encoder'] += param.numel()
        elif name.startswith('pooler'):
            groups['pooler'] += param.numel()
        else:
            groups['others'] += param.numel()
    print("\nGrouped by module:")
    for k, v in groups.items():
        print(f"{k:12}: {v:,}")

# 加载BERT模型
model = BertModel.from_pretrained('bert-base-chinese')

# 详细打印每个参数
print_num_params(model)

# 按大模块分组统计
print_grouped_params(model)
