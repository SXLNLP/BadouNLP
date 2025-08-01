import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model=BertModel.from_pretrained(r"D:/Code/py/八斗nlp/20250622/week6 语言模型和预训练/bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hidden_size = 3072            # 隐藏层维数
num_layers = 1             # 隐藏层层数

#embedding层参数 v*h+n*h+max_sequence_length*h+layernorm层参数
# embedding_size + embedding_sizes是layer_norm层参数
embedding_params=vocab*embedding_size+n*embedding_size+max_sequence_length*embedding_size+embedding_size + embedding_size

#self_attention层参数
self_attention_params=(embedding_size*embedding_size+embedding_size)*3

#self_attention_out参数：“多头注意力后的输出子层”（attention output sub-layer）里的全部可学习参数
#embedding_size * embedding_size + embedding_size是self输出的线性层参数，embedding_size + embedding_size是layer_norm层参数
self_attention_out_params=(embedding_size * embedding_size + embedding_size) + embedding_size + embedding_size

#feed_forward参数   2个线性层y=kx+b
# embedding_size * hide_size + hide_size第一个线性层，embedding_size * hide_size + embedding_size第二个线性层，
# embedding_size + embedding_size是layer_norm层
feed_forward_params=(embedding_size * hidden_size + hidden_size )+ (embedding_size * hidden_size + embedding_size) + embedding_size + embedding_size

#pooler参数
pool_params=embedding_size * embedding_size + embedding_size

# 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
all_paramerters = embedding_params + (self_attention_params + self_attention_out_params +feed_forward_params) * num_layers + pool_params                                                                                                                                                                 
print("实际参数%d" % sum(p.numel() for p in model.parameters()))
print("diy参数%d" % all_paramerters)
