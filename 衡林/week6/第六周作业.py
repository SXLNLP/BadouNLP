import torch
from transformers import BertModel
import json

# 1. 读取BERT模型的配置文件
config_file_path = "./bert-base-chinese/config.json"

# 打开配置文件
with open(config_file_path, 'r', encoding='utf-8') as config_file:
    # 加载JSON配置
    bert_config = json.load(config_file)

# 从配置中获取各种参数
vocab_size = bert_config['vocab_size']  # 词汇表大小
max_position = bert_config['max_position_embeddings']  # 最大位置编码长度
type_vocab_size = bert_config['type_vocab_size']  # 句子类型数量
hidden_size = bert_config['hidden_size']  # 隐藏层大小
intermediate_size = bert_config['intermediate_size']  # 中间层大小
num_attention_heads = bert_config['num_attention_heads']  # 注意力头数量
num_hidden_layers = bert_config['num_hidden_layers']  # 隐藏层数量

# 2. 计算各部分参数数量

# 计算Embedding层的参数数量
def calculate_embedding_params():
    # 词嵌入参数 = 词汇表大小 × 隐藏层大小
    word_embedding_params = vocab_size * hidden_size
    
    # 位置嵌入参数 = 最大位置 × 隐藏层大小
    position_embedding_params = max_position * hidden_size
    
    # 句子类型嵌入参数 = 句子类型数量 × 隐藏层大小
    token_type_embedding_params = type_vocab_size * hidden_size
    
    # 层归一化参数 = 2 × 隐藏层大小 (权重和偏置)
    layer_norm_params = 2 * hidden_size
    
    # 总参数 = 所有部分相加
    total_embedding_params = (word_embedding_params + position_embedding_params 
                             + token_type_embedding_params + layer_norm_params)
    
    return total_embedding_params

# 计算自注意力层的参数数量
def calculate_self_attention_params():
    # 查询(Q)投影层的参数 = 隐藏层 × 隐藏层 (权重) + 隐藏层 (偏置)
    query_proj_params = hidden_size * hidden_size + hidden_size
    
    # 键(K)投影层的参数
    key_proj_params = hidden_size * hidden_size + hidden_size
    
    # 值(V)投影层的参数
    value_proj_params = hidden_size * hidden_size + hidden_size
    
    # 输出投影层的参数
    output_proj_params = hidden_size * hidden_size + hidden_size
    
    # 注意力输出后的层归一化参数
    attention_layer_norm_params = 2 * hidden_size
    
    # 总参数 = 所有部分相加
    total_attention_params = (query_proj_params + key_proj_params 
                             + value_proj_params + output_proj_params 
                             + attention_layer_norm_params)
    
    return total_attention_params

# 计算前馈神经网络的参数数量
def calculate_feed_forward_params():
    # 中间层参数 = 隐藏层 × 中间层大小 + 中间层大小 (偏置)
    intermediate_proj_params = hidden_size * intermediate_size + intermediate_size
    
    # 输出层参数 = 中间层大小 × 隐藏层 + 隐藏层 (偏置)
    output_proj_params = intermediate_size * hidden_size + hidden_size
    
    # 输出层归一化参数
    output_layer_norm_params = 2 * hidden_size
    
    # 总参数 = 所有部分相加
    total_feed_forward_params = (intermediate_proj_params + output_proj_params 
                                + output_layer_norm_params)
    
    return total_feed_forward_params

# 计算单个Transformer层的参数数量
def calculate_transformer_layer_params():
    # Transformer层 = 自注意力层 + 前馈神经网络
    attention_params = calculate_self_attention_params()
    feed_forward_params = calculate_feed_forward_params()
    
    return attention_params + feed_forward_params

# 计算Pooler层的参数数量
def calculate_pooler_params():
    # Pooler层 = 隐藏层 × 隐藏层 (权重) + 隐藏层 (偏置)
    return hidden_size * hidden_size + hidden_size

# 3. 计算BERT模型的总参数

# 计算各部分参数
embedding_params = calculate_embedding_params()
single_layer_params = calculate_transformer_layer_params()
all_layers_params = single_layer_params * num_hidden_layers
pooler_params = calculate_pooler_params()

# 总参数 = 嵌入层 + 所有Transformer层 + Pooler层
total_bert_params = embedding_params + all_layers_params + pooler_params

# 4. 打印计算结果
print("=== 手动计算的BERT模型参数 ===")
print(f"嵌入层参数数量: {embedding_params:,}")
print(f"单个Transformer层参数数量: {single_layer_params:,}")
print(f"所有Transformer层参数数量: {all_layers_params:,}")
print(f"Pooler层参数数量: {pooler_params:,}")
print(f"BERT模型总参数数量: {total_bert_params:,}")
print()

# 5. 加载实际BERT模型并统计参数

# 加载预训练的BERT模型
bert_model = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
bert_model.eval()  # 设置为评估模式

# 计算模型总参数
def count_total_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()  # numel()返回参数数量
    return total

# 详细统计各部分参数
def detailed_param_count(model):
    param_details = {}
    
    # 统计嵌入层参数
    embedding_total = 0
    # 词嵌入
    embedding_total += model.embeddings.word_embeddings.weight.numel()
    # 位置嵌入
    embedding_total += model.embeddings.position_embeddings.weight.numel()
    # 句子类型嵌入
    embedding_total += model.embeddings.token_type_embeddings.weight.numel()
    # 层归一化
    embedding_total += model.embeddings.LayerNorm.weight.numel()
    embedding_total += model.embeddings.LayerNorm.bias.numel()
    
    param_details['embeddings'] = embedding_total
    
    # 统计编码器(Transformer)参数
    encoder_total = 0
    layer_count = len(model.encoder.layer)
    
    for layer in model.encoder.layer:
        layer_params = 0
        # 自注意力部分
        layer_params += layer.attention.self.query.weight.numel()
        layer_params += layer.attention.self.query.bias.numel()
        layer_params += layer.attention.self.key.weight.numel()
        layer_params += layer.attention.self.key.bias.numel()
        layer_params += layer.attention.self.value.weight.numel()
        layer_params += layer.attention.self.value.bias.numel()
        layer_params += layer.attention.output.dense.weight.numel()
        layer_params += layer.attention.output.dense.bias.numel()
        layer_params += layer.attention.output.LayerNorm.weight.numel()
        layer_params += layer.attention.output.LayerNorm.bias.numel()
        
        # 前馈神经网络部分
        layer_params += layer.intermediate.dense.weight.numel()
        layer_params += layer.intermediate.dense.bias.numel()
        layer_params += layer.output.dense.weight.numel()
        layer_params += layer.output.dense.bias.numel()
        layer_params += layer.output.LayerNorm.weight.numel()
        layer_params += layer.output.LayerNorm.bias.numel()
        
        encoder_total += layer_params
    
    # 计算单层平均参数
    param_details['encoder_layer_avg'] = int(encoder_total / layer_count)
    param_details['encoder'] = encoder_total
    
    # 统计Pooler层参数
    pooler_total = 0
    pooler_total += model.pooler.dense.weight.numel()
    pooler_total += model.pooler.dense.bias.numel()
    param_details['pooler'] = pooler_total
    
    return param_details

# 获取详细参数统计
detailed_params = detailed_param_count(bert_model)
model_total_params = count_total_params(bert_model)

# 打印实际模型参数统计
print("=== 实际BERT模型的参数统计 ===")
print(f"BERT模型总参数数量: {model_total_params:,}")
print(f"嵌入层参数数量: {detailed_params['embeddings']:,}")
print(f"Transformer编码器总参数数量: {detailed_params['encoder']:,}")
print(f"单个Transformer层平均参数数量: {detailed_params['encoder_layer_avg']:,}")
print(f"Pooler层参数数量: {detailed_params['pooler']:,}")
