import math

hidden_size = 768
hidden_layers = 12
intermediate_size = 3072
vocab_size = 21128
max_embeddings = 512
type_vocab_size = 2

def single_transformer_layer():
    attention_weight = 4 * hidden_size * hidden_size
    attention_bias = 4 * hidden_size
    ffn_weight = 2 * hidden_size * intermediate_size
    ffn_bias = intermediate_size + hidden_size
    layernorm = 2 * hidden_size * 2
    return attention_weight + attention_bias + ffn_weight + ffn_bias + layernorm

def calculate_bert_params():
    embedding_params = ((vocab_size + max_embeddings + type_vocab_size) * hidden_size + 2 * hidden_size)
    encoder_params = hidden_layers * single_transformer_layer()
    pooler_params = hidden_size * hidden_size + hidden_size
    total_params = embedding_params + encoder_params + pooler_params
    print("\n BERT 总参数量: {:,} (~{:.2f}M)".format(total_params, total_params / 1e6))
    return total_params

calculate_bert_params()
