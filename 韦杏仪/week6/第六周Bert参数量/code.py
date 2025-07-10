import torch
import math
import numpy as np
from transformers import BertModel

'''
通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models
'''
bert = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

#定义Bert模型的基本配置参数
vocab_size = 30522  #词表大小
max_position = 512  #文本输入最长大小
type_vocab_size = 2 #句子类型数量
hidden_size = 768   #隐藏层大小
num_layers = 12     #Transformer层数
num_attention_heads = 12  #注意力头数量

#计算嵌入层的参数量
token_embedding_params = vocab_size * hidden_size
position_embedding_params = max_position * hidden_size
type_embedding_params = type_vocab_size * hidden_size

embedding_norm_params = hidden_size * 2

#计算嵌入层总参数量
total_embedding_params = token_embedding_params + position_embedding_params + type_embedding_params + embedding_norm_params

#计算单个注意力头的参数量
attention_head_size = hidden_size // num_attention_heads

#计算单个Transformer层的参数量
qkv_weights = 3 * hidden_size * hidden_size  # QKV矩阵权重
qkv_bias = 3 * hidden_size  # QKV矩阵偏置
attention_weights = qkv_weights + qkv_bias

output_weights = hidden_size * hidden_size
output_bias = hidden_size
attention_output_params = output_weights + output_bias

intermediate_size = hidden_size * 4  # 中间层维度
ffn_weights_1 = hidden_size * intermediate_size
ffn_bias_1 = intermediate_size
ffn_weights_2 = intermediate_size * hidden_size
ffn_bias_2 = hidden_size
ffn_params = ffn_weights_1 + ffn_bias_1 + ffn_weights_2 + ffn_bias_2

norm_params = 2 * hidden_size * 2  # 两个层归一化层

single_layer_params = attention_weights + attention_output_params + ffn_params + norm_params
all_transformer_params = single_layer_params * num_layers

#pooler层参数量
pooler_weights = hidden_size * hidden_size
pooler_bias = hidden_size
pooler_params = pooler_weights + pooler_bias

#计算Bert模型的总参数量
total_params = total_embedding_params + all_transformer_params + pooler_params

print("Bert模型参数量计算结果：")
print(f"嵌入层参数: {total_embedding_params:,}")
print(f"Transformer层参数: {all_transformer_params:,}")
print(f"pooler层参数: {pooler_params:,}")
print(f"Bert总参数量: {total_params:,}")
