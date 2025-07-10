import json

# 使用完整文件路径打开文件
file_path = r'H:\AI\AI 八斗\week6 语言模型\bert-base-chinese\config.json'
with open(file_path, 'r') as f:
    config = json.load(f)

# 提取必要的参数
hidden_size = config["hidden_size"]
num_attention_heads = config["num_attention_heads"]
num_hidden_layers = config["num_hidden_layers"]
vocab_size = config["vocab_size"]
max_position_embeddings = config["max_position_embeddings"]
type_vocab_size = config["type_vocab_size"]
pooler_fc_size = config["pooler_fc_size"]
pooler_num_fc_layers = config["pooler_num_fc_layers"]
num_labels = config["num_labels"]

# 计算 Self-attention 部分的总参数量
# 每个注意力头的输入和输出维度
head_size = hidden_size // num_attention_heads
# 每个注意力头的 Q, K, V 矩阵参数
single_head_qkv_params = head_size * hidden_size * 3
# 每个注意力头的输出投影矩阵参数
single_head_output_params = head_size * hidden_size
# 单个 Self-attention 层的总参数量
single_self_attention_params = (single_head_qkv_params + single_head_output_params) * num_attention_heads
# Self-attention 部分的总参数量（所有层）
self_attention_total_params = single_self_attention_params * num_hidden_layers

# 计算 Multi-Head Attention 部分的总参数量
# Multi-Head Attention 和 Self-attention 参数量计算方式相同
multi_head_attention_total_params = self_attention_total_params

# 计算 Embedding 层的总参数量
# 词嵌入层参数
word_embeddings_params = vocab_size * hidden_size
# 位置嵌入层参数
position_embeddings_params = max_position_embeddings * hidden_size
# 类型嵌入层参数
type_embeddings_params = type_vocab_size * hidden_size
# 嵌入层归一化参数
embedding_layer_norm_params = hidden_size * 2
# 嵌入层总参数
embedding_total_params = word_embeddings_params + position_embeddings_params + type_embeddings_params + embedding_layer_norm_params

# 每个隐藏层的 FFN 参数
intermediate_size = config["intermediate_size"]
single_ffn_params = hidden_size * intermediate_size + intermediate_size + intermediate_size * hidden_size + hidden_size
# 所有隐藏层的 FFN 参数
ffn_total_params = single_ffn_params * num_hidden_layers

# 池化层参数
pooler_params = pooler_fc_size * hidden_size + pooler_fc_size
# 分类器参数
classifier_params = pooler_fc_size * num_labels + num_labels

# BERT 模型的总参数量
bert_total_params = embedding_total_params + self_attention_total_params + ffn_total_params + pooler_params + classifier_params

print(f"Self-attention 部分的总参数量: {self_attention_total_params}")
print(f"Multi-Head Attention 部分的总参数量: {multi_head_attention_total_params}")
print(f"Embedding 层的总参数量: {embedding_total_params}")
print(f"BERT 模型的总参数量: {bert_total_params}")
