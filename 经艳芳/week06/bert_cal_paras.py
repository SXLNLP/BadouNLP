import json

def calculate_bert_parameters(config_path="config.json"):
    with open(config_path) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    type_vocab_size = config["type_vocab_size"]
    max_position_embeddings = config["max_position_embeddings"]
    num_hidden_layers = config["num_hidden_layers"]
    intermediate_size = config["intermediate_size"]

    print("--- 模型配置 ---")
    print(f"隐藏层维度 (hidden_size): {hidden_size}")
    print(f"词汇表大小 (vocab_size): {vocab_size}")
    print(f"Encoder 层数 (num_hidden_layers): {num_hidden_layers}")
    print(f"前馈网络中间层维度 (intermediate_size): {intermediate_size}")
    print("-" * 30)

    # Embedding 层参数

    token_embed_params = vocab_size * hidden_size
    position_embed_params = max_position_embeddings * hidden_size
    segment_embed_params = type_vocab_size * hidden_size

    embed_layer_norm_params = 2 * hidden_size

    total_embedding_params = (
            token_embed_params +
            position_embed_params +
            segment_embed_params +
            embed_layer_norm_params
    )

    # Encoder 层参数

    # 单个 Encoder 层的自注意力 (Self-Attention)
    qkv_weights = 3 * (hidden_size * hidden_size)
    qkv_biases = 3 * hidden_size
    attention_output_weights = hidden_size * hidden_size
    attention_output_biases = hidden_size
    # Layer Normalization: 在残差连接后进行归一化
    attention_layer_norm_params = 2 * hidden_size

    single_attention_params = (
            qkv_weights + qkv_biases +
            attention_output_weights + attention_output_biases +
            attention_layer_norm_params
    )

    # 单个 Encoder 层的前馈网络 (Feed-Forward Network)
    intermediate_weights = hidden_size * intermediate_size
    intermediate_biases = intermediate_size

    ffn_output_weights = intermediate_size * hidden_size
    ffn_output_biases = hidden_size
    # Layer Normalization: 在残差连接后进行归一化
    ffn_layer_norm_params = 2 * hidden_size

    single_ffn_params = (
            intermediate_weights + intermediate_biases +
            ffn_output_weights + ffn_output_biases +
            ffn_layer_norm_params
    )

    single_encoder_layer_params = single_attention_params + single_ffn_params

    total_encoder_params = single_encoder_layer_params * num_hidden_layers

    # Pooler 层参数
    pooler_weights = hidden_size * hidden_size
    pooler_biases = hidden_size

    total_pooler_params = pooler_weights + pooler_biases


    total_params = (
            total_embedding_params +
            total_encoder_params +
            total_pooler_params
    )

    print("--- 参数量分解 ---")
    print(f"Embedding 层:       {total_embedding_params:12,}")
    print(f"Encoder 层 (单层):    {single_encoder_layer_params:12,}")
    print(f"  - 自注意力:  {single_attention_params:12,}")
    print(f"  - 前馈网络:  {single_ffn_params:12,}")
    print(f"Encoder 层 (共 {num_hidden_layers} 层): {total_encoder_params:12,}")
    print(f"Pooler 层:          {total_pooler_params:12,}")
    print("-" * 30)
    print(f"模型总参数量:       {total_params:12,}")
    print(f"模型总参数量 (百万 M): {total_params / 1_000_000:.2f}M")

    return total_params

if __name__ == "__main__":
    calculate_bert_parameters("config.json")
