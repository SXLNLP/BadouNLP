import torch
import math
import numpy as np
from transformers import BertModel

"""
计算bert模型所有参数的总量
"""

bert = BertModel.from_pretrained(r"D:\workspace\pynlp_workspace\ycy2025\model\bert-base-chinese", return_dict=False)
state = bert.state_dict()

bert_float = 0
for weight_name in bert.state_dict().keys():
    print(weight_name)
    # if weight_name == "embeddings.word_embeddings.weight":

    print(bert.state_dict()[weight_name].shape)
    if len(bert.state_dict()[weight_name].shape) == 2:
        h, w = bert.state_dict()[weight_name].shape
        bert_float += h * w
    else:
        s = bert.state_dict()[weight_name].shape
        bert_float += s[0]
print("bert_float:", bert_float)

# 字表层
word_embeddings = 21128 * 768
# 句型层
token_type_embeddings = 2 * 768
# 位置层
position_embeddings = 512 * 768
# embedding归一化权重
LayerNorm_weight = 768
LayerNorm_bias = 768
# transformer层
# Q矩阵权重
query_weight = 768 * 768
query_bias = 768
# K矩阵权重
key_weight = 768 * 768
key_bias = 768
# V矩阵权重
value_weight = 768 * 768
value_bias = 768
# 多头机制拼接后自注意力层线性层矩阵权重
attention_output_dense_weight = 768 * 768
attention_output_dense_bias = 768
# 残差机制中归一化层的权重
attention_output_LayerNorm_weight = 768
attention_output_LayerNorm_bias = 768
# feed forward接收残差输出的线性层权重
intermediate_dense_weight = 3072 * 768
intermediate_dense_bias = 3072
# feed forward输出层线性层权重
output_dense_weight = 768 * 3072
output_dense_bias = 768
# feed forward残差机制中归一化层的权重
output_LayerNorm_weight = 768
output_LayerNorm_bias = 768
# 句向量生成线性层权重
pooler_dense_weight = 768 * 768
pooler_dense_bias = 768

bert_weight = word_embeddings + token_type_embeddings + position_embeddings + LayerNorm_weight + LayerNorm_bias + 12 * (query_weight + query_bias + key_weight + key_bias + value_weight + value_bias + attention_output_dense_weight + attention_output_dense_bias + attention_output_LayerNorm_weight + attention_output_LayerNorm_bias + intermediate_dense_weight + intermediate_dense_bias + output_dense_weight + output_dense_bias + output_LayerNorm_weight + output_LayerNorm_bias) + pooler_dense_weight + pooler_dense_bias
print("bert_weight:",bert_weight)
