import torch
from transformers import BertModel, BertConfig
import os

def count_parameters(model):
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters())

def print_parameters_by_layer(model):
    """打印模型每一层的参数量"""
    total_params = 0
    # 打印每个命名参数的形状和参数量
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: {param.shape} - {param_count:,} 参数")
    
    # 打印总参数量
    print(f"\n总参数量: {total_params:,}")
    return total_params

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bert_path = os.path.join(current_dir, "bert-base-chinese")
    
    # 方法1: 使用预训练的BERT模型
    print("加载预训练的BERT模型...")
    try:
        bert = BertModel.from_pretrained(bert_path)
        print("\n预训练的BERT模型参数统计:")
        print_parameters_by_layer(bert)
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        print("尝试使用BERT配置创建模型...")
    
    # 方法2: 使用BERT配置创建模型
    print("\n\n使用BERT-base配置创建模型...")
    # BERT-base配置
    config = BertConfig(
        vocab_size=21128,  # 中文词表大小
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    model = BertModel(config)
    print("\nBERT-base模型参数统计:")
    total = print_parameters_by_layer(model)
    
    # 打印BERT模型主要组成部分的参数量
    print("\nBERT模型主要组成部分的参数量:")
    
    # 嵌入层参数量
    embedding_params = (
        config.vocab_size * config.hidden_size +  # 词嵌入
        config.max_position_embeddings * config.hidden_size +  # 位置嵌入
        config.type_vocab_size * config.hidden_size  # 类型嵌入
    )
    print(f"嵌入层参数量: {embedding_params:,}")
    
    # 每个Transformer层参数量
    attention_params_per_layer = (
        4 * config.hidden_size * config.hidden_size +  # Q, K, V 和输出投影
        4 * config.hidden_size  # 偏置项
    )
    
    ff_params_per_layer = (
        config.hidden_size * config.intermediate_size +  # 第一个全连接层
        config.intermediate_size +  # 第一个全连接层偏置
        config.intermediate_size * config.hidden_size +  # 第二个全连接层
        config.hidden_size  # 第二个全连接层偏置
    )
    
    layer_norm_params = 4 * config.hidden_size  # 两个LayerNorm层(每层2个参数)
    
    transformer_layer_params = attention_params_per_layer + ff_params_per_layer + layer_norm_params
    total_transformer_params = transformer_layer_params * config.num_hidden_layers
    
    print(f"每个Transformer层参数量: {transformer_layer_params:,}")
    print(f"所有Transformer层参数量: {total_transformer_params:,}")
    
    # Pooler层参数量
    pooler_params = config.hidden_size * config.hidden_size + config.hidden_size
    print(f"Pooler层参数量: {pooler_params:,}")
    
    # 理论计算的总参数量
    theoretical_total = embedding_params + total_transformer_params + pooler_params
    print(f"理论计算的总参数量: {theoretical_total:,}")
    print(f"实际统计的总参数量: {total:,}")

if __name__ == "__main__":
    main()
