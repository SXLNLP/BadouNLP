import torch
from transformers import BertModel

# 请务必确认使用纯英文路径
model = BertModel.from_pretrained(r"E:\代码答案\week06答案\week6\bert-base-chinese", return_dict=False)

vocab = 21128
max_sequence_length = 512
embedding_size = 768
intermediate_size = 3072  # FFN隐藏层大小
num_layers = 12           # 关键修正：12层
num_attention_heads = 12  # 注意力头数

# 1. Embedding层
embedding_params = (
    vocab * embedding_size +        # token embedding
    max_sequence_length * embedding_size +  # position embedding
    2 * embedding_size +            # segment embedding (2种句子类型)
    2 * embedding_size               # LayerNorm
)

# 2. 每层编码器参数
per_layer_params = 0
# 自注意力部分 (Q,K,V投影)
per_layer_params += 3 * (embedding_size * embedding_size + embedding_size)
# 自注意力输出
per_layer_params += (embedding_size * embedding_size + embedding_size)  # 线性层
per_layer_params += 2 * embedding_size  # LayerNorm
# 前馈神经网络
per_layer_params += (embedding_size * intermediate_size + intermediate_size)  # 第一层
per_layer_params += (intermediate_size * embedding_size + embedding_size)  # 第二层
per_layer_params += 2 * embedding_size  # LayerNorm

# 3. 池化层
pooler_params = embedding_size * embedding_size + embedding_size

# 总参数计算
total_params = (
    embedding_params +
    per_layer_params * num_layers +
    pooler_params
)

# 实际参数统计
actual_params = sum(p.numel() for p in model.parameters())

print(f"模型实际参数: {actual_params}")
print(f"修正计算参数: {total_params}")
print(f"差异百分比: {abs(actual_params - total_params)/actual_params:.2%}")
