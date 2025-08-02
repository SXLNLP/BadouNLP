import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

# model = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数
num_layers = 1             # 隐藏层层数

# embedding过程中的参数，其中 vocab * embedding_size是词表embedding参数， max_sequence_length * embedding_size是位置参数， n * embedding_size是句子参数
# embedding_size + embedding_sizes是layer_norm层参数
embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

# self_attention过程的参数, 其中embedding_size * embedding_size是权重参数，embedding_size是bias， *3是K Q V三个
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out参数 其中 embedding_size * embedding_size + embedding_size 是self输出的线性层参数，embedding_size + embedding_size是layer_norm层参数
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# Feed Forward参数 其中embedding_size * hide_size + hide_size第一个线性层，embedding_size * hide_size + embedding_size第二个线性层，
# embedding_size + embedding_size是layer_norm层
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

# pool_fc层参数
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
all_paramerters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters) * num_layers + pool_fc_parameters

print("diy计算参数个数为%d" % all_paramerters)

# 1. 
embedding =  vocab * embedding_size + max_sequence_length * embedding_size + 2* embedding_size + embedding_size + embedding_size
# 2.
self_attention_parameters = 3 * (embedding_size * embedding_size + embedding_size)
# 3
self_attention_out =  embedding_size * embedding_size + embedding_size + ( embedding_size + embedding_size)
# 4.
feed_forward_parameters =  (embedding_size * embedding_size * 4  + 4 * embedding_size) + (4 * embedding_size * embedding_size + embedding_size) + (embedding_size + embedding_size)

# 5.
pool_fc_parameters = embedding_size * embedding_size + embedding_size

print(embedding + self_attention_parameters + self_attention_out + feed_forward_parameters + pool_fc_parameters)