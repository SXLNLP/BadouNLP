hidden_size = 768
num_hidden_layers = 12
intermediate_size = 3072
vocab_size = 21128
max_position_embeddings = 512
type_vocab_size = 2

def calculate_bert_params(
):
    # 1. Embedding 层（Token + Position + Segment）+ LayerNorm
    embedding_params = (
        (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size +
        2 * hidden_size  # LayerNorm gamma + beta
)

    # 2. 单层 Transformer 参数
    def single_transformer_layer():
        # 自注意力层（Q, K, V, Output）
        attention_weight = 4 * hidden_size * hidden_size
        attention_bias = 4 * hidden_size

        # 前馈网络：
        ffn_weight = 2 * hidden_size * intermediate_size
        ffn_bias = intermediate_size + hidden_size

        #归一化 ×2
        layernorm = 2 * hidden_size * 2

        return attention_weight + attention_bias + ffn_weight + ffn_bias + layernorm

    # 3. Transformer 编码器总参数
    encoder_params = num_hidden_layers * single_transformer_layer()

    # 4. Pooler 层
    pooler_params = hidden_size * hidden_size + hidden_size


    # 总计
    total_params = embedding_params + encoder_params + pooler_params

    print(f"\n BERT 总参数量: {total_params:,} (~{total_params / 1e6:.2f}M)")

    return total_params


calculate_bert_params()
