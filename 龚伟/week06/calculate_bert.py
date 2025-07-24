import torch
from transformers import BertModel

model = BertModel.from_pretrained(
    r"D:/Code/py/八斗nlp/20250622/week6 语言模型和预训练/bert-base-chinese",
    return_dict=False
)

# 直接从 config 里读真实超参数，避免手写出错
cfg = model.config
vocab              = cfg.vocab_size            # 21128
embedding_size     = cfg.hidden_size           # 768
max_sequence_length= cfg.max_position_embeddings  # 512
token_type_vocab   = cfg.type_vocab_size       # 2
hidden_size        = cfg.intermediate_size     # 3072
num_layers         = cfg.num_hidden_layers     # 12   <-- 这里就是 12！

# 1. Embedding
embedding_params = (
    vocab              * embedding_size +      # word
    max_sequence_length* embedding_size +      # position
    token_type_vocab   * embedding_size +      # segment
    2 * embedding_size                         # LayerNorm γ + β
)

# 2. Multi-head Self-Attention
self_attention_params = 3 * (embedding_size * embedding_size + embedding_size)

# 3. Attention output + LayerNorm
self_attention_out_params = (
    embedding_size * embedding_size +          # W_O
    embedding_size +                           # b_O
    2 * embedding_size                         # LayerNorm γ + β
)

# 4. Feed-Forward
ffn_params = (
    embedding_size * hidden_size + hidden_size +   # intermediate W + b
    hidden_size  * embedding_size + embedding_size +  # output W + b
    2 * embedding_size                              # LayerNorm γ + β
)

# 5. Pooler
pool_params = embedding_size * embedding_size + embedding_size

# 6. 总计
transformer_block_params = (
    self_attention_params +
    self_attention_out_params +
    ffn_params
)

total_manual = embedding_params + num_layers * transformer_block_params + pool_params
total_actual = sum(p.numel() for p in model.parameters())

print(f"手工计算: {total_manual:,}")
print(f"PyTorch统计: {total_actual:,}")
