import math

def calculate_bert_params(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=30522,
        max_position_embeddings=512,
        type_vocab_size=2
):
    # 计算词嵌入层参数
    word_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    token_type_embeddings = type_vocab_size * hidden_size
    embedding_params = word_embeddings + position_embeddings + token_type_embeddings

    # 计算每个Transformer层的参数
    # 注意力机制参数
    qkv_params = 3 * hidden_size * hidden_size
    output_params = hidden_size * hidden_size
    attention_params = qkv_params + output_params

    # 前馈网络参数
    intermediate_params = hidden_size * intermediate_size
    output_params = intermediate_size * hidden_size
    feed_forward_params = intermediate_params + output_params

    # 层归一化和偏置参数
    layer_norm_params = 2 * hidden_size * 2  # 两个层归一化

    # 单个Transformer层总参数
    single_layer_params = attention_params + feed_forward_params + layer_norm_params

    # 所有Transformer层参数
    transformer_params = single_layer_params * num_hidden_layers

    # 池化层参数
    pooler_params = hidden_size * hidden_size

    # 总参数
    total_params = embedding_params + transformer_params + pooler_params

    return {
        "embedding_params": embedding_params,
        "transformer_params": transformer_params,
        "pooler_params": pooler_params,
        "total_params": total_params
    }


# 计算BERT-base模型参数量
params = calculate_bert_params()
print(f"BERT-base模型总参数量: {params['total_params']:,}")
print(f"其中:")
print(f"- 嵌入层参数: {params['embedding_params']:,}")
print(f"- Transformer层参数: {params['transformer_params']:,}")
print(f"- 池化层参数: {params['pooler_params']:,}")
