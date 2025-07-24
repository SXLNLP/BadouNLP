import torch
from transformers import BertModel

def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_module_parameters(model, module_name):
    """计算模型中特定模块的参数量"""
    module_params = 0
    for name, param in model.named_parameters():
        if module_name in name:
            module_params += param.numel()
    return module_params

def count_attention_params(model, layer_idx):
    """精确统计指定层的注意力机制参数量"""
    prefix = f"encoder.layer.{layer_idx}.attention"
    params = 0
    
    # 注意力QKV矩阵参数
    for mat_type in ["query", "key", "value"]:
        # 权重矩阵
        params += count_module_parameters(model, f"{prefix}.self.{mat_type}.weight")
        # 偏置向量
        params += count_module_parameters(model, f"{prefix}.self.{mat_type}.bias")
    
    # 注意力输出投影参数
    params += count_module_parameters(model, f"{prefix}.output.dense.weight")
    params += count_module_parameters(model, f"{prefix}.output.dense.bias")
    
    # 层归一化参数 (可选：是否包含在注意力机制中)
    params += count_module_parameters(model, f"{prefix}.output.LayerNorm.weight")
    params += count_module_parameters(model, f"{prefix}.output.LayerNorm.bias")
    
    return params

def print_bert_parameter_stats(model):
    """打印BERT模型的参数量统计信息"""
    total_params, trainable_params = count_parameters(model)
    
    # 分类统计各部分参数量
    embedding_params = count_module_parameters(model, "embeddings")
    encoder_params = count_module_parameters(model, "encoder")
    attention_params = count_module_parameters(model, "attention")
    feedforward_params = count_module_parameters(model, "intermediate") + count_module_parameters(model, "output")
    pooler_params = count_module_parameters(model, "pooler")
    
    print("=" * 50)
    print(f"BERT模型参数统计")
    print("=" * 50)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("\n各组件参数量:")
    print(f"  1. 嵌入层: {embedding_params:,} ({embedding_params/total_params*100:.2f}%)")
    print(f"  2. 编码器: {encoder_params:,} ({encoder_params/total_params*100:.2f}%)")
    print(f"    - 注意力机制: {attention_params:,} ({attention_params/encoder_params*100:.2f}%)")
    print(f"    - 前馈网络: {feedforward_params:,} ({feedforward_params/encoder_params*100:.2f}%)")
    print(f"  3. Pooler层: {pooler_params:,} ({pooler_params/total_params*100:.2f}%)")
    print("=" * 50)

# 加载BERT模型
model = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese", return_dict=False)

# 打印参数量统计
print_bert_parameter_stats(model)

# 按层打印注意力头的参数量
print("\n每层注意力机制参数量:")
for i in range(model.config.num_hidden_layers):
    attn_params = count_attention_params(model, i)
    print(f"  第 {i+1} 层: {attn_params:,}")
