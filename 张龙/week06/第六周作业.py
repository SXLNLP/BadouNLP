import torch
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-chinese", return_dict=False)

# 模型超参数
vocab = 21128                # 词表大小
n = 2                        # 输入最大句子个数（如两个句子拼接做句子对任务）
max_sequence_length = 512    # 最大序列长度
embedding_size = 768         # embedding维度
hide_size = 3072             # 前馈神经网络隐藏层大小
num_layers = 12              # Transformer层数

# 1. Embedding层参数量
# 含词嵌入、位置嵌入、token类型嵌入，再加一个LayerNorm
embedding_params = vocab * embedding_size + \
                   max_sequence_length * embedding_size + \
                   n * embedding_size + \
                   embedding_size

# 2. Self-Attention参数量（每层都有）
# 每一层的注意力机制：W_q、W_k、W_v 权重和 bias，总共 3 组
self_attention_params = num_layers * (embedding_size * embedding_size + embedding_size) * 3

# 3. Self-Attention输出部分（线性变换 + LayerNorm）
self_attention_out_params = num_layers * (embedding_size * embedding_size + embedding_size + embedding_size)

# 4. Feed Forward层参数量（包含两个线性层 + bias + LayerNorm）
feed_forward_params = num_layers * (
    embedding_size * hide_size + hide_size +  # 第一层线性层及其bias
    hide_size * embedding_size + embedding_size +  # 第二层线性层及其bias
    embedding_size  # LayerNorm
)

# 总参数量估算
total_params = embedding_params + self_attention_params + self_attention_out_params + feed_forward_params

# 打印各部分和总参数量
print(f"Embedding 层参数量: {embedding_params}")
print(f"Self-Attention 层参数量: {self_attention_params}")
print(f"Self-Attention 输出参数量: {self_attention_out_params}")
print(f"Feed Forward 层参数量: {feed_forward_params}")
print(f"总参数量估算: {total_params}")
