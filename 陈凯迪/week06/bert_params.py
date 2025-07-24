import json

# 加载配置文件
config = {
    "architectures": ["BertForMaskedLM"],
    "hidden_act": "gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 21128
}


def calculate_bert_parameters(config):
    # 1. 嵌入层参数
    word_embeddings = config["vocab_size"] * config["hidden_size"]
    position_embeddings = config["max_position_embeddings"] * config["hidden_size"]
    token_type_embeddings = config["type_vocab_size"] * config["hidden_size"]
    embeddings_layer_norm = 2 * config["hidden_size"]  # gamma + beta

    embeddings_total = (
            word_embeddings +
            position_embeddings +
            token_type_embeddings +
            embeddings_layer_norm
    )

    # 2. Transformer层参数（每层）
    def transformer_layer_params():
        # 注意力层参数
        qkv_weight = 3 * config["hidden_size"] * config["hidden_size"]
        qkv_bias = 3 * config["hidden_size"]

        # 注意力输出层
        attention_output_weight = config["hidden_size"] * config["hidden_size"]
        attention_output_bias = config["hidden_size"]

        # 注意力层归一化
        attention_layer_norm = 2 * config["hidden_size"]  # gamma + beta

        # 前馈网络（全连接层）
        intermediate_weight = config["hidden_size"] * config["intermediate_size"]
        intermediate_bias = config["intermediate_size"]

        # 输出层
        output_weight = config["intermediate_size"] * config["hidden_size"]
        output_bias = config["hidden_size"]

        # 前馈层归一化
        ff_layer_norm = 2 * config["hidden_size"]  # gamma + beta

        return (
                qkv_weight + qkv_bias +
                attention_output_weight + attention_output_bias +
                attention_layer_norm +
                intermediate_weight + intermediate_bias +
                output_weight + output_bias +
                ff_layer_norm
        )

    # 3. 池化层参数
    pooler_weight = config["hidden_size"] * config["hidden_size"]
    pooler_bias = config["hidden_size"]
    pooler_total = pooler_weight + pooler_bias

    # 4. 总参数计算
    transformer_layers = config["num_hidden_layers"] * transformer_layer_params()
    total_params = embeddings_total + transformer_layers + pooler_total

    # 5. MLM头部参数（用于掩码语言建模）
    mlm_weight = config["hidden_size"] * config["vocab_size"]
    mlm_bias = config["vocab_size"]
    mlm_total = mlm_weight + mlm_bias

    # 包含MLM头的总参数
    total_with_mlm = total_params + mlm_total

    # 返回计算结果
    return {
        "embeddings": embeddings_total,
        "transformer_layers": transformer_layers,
        "pooler": pooler_total,
        "mlm_head": mlm_total,
        "bert_base_total": total_params,
        "bert_mlm_total": total_with_mlm
    }


# 计算参数量
params = calculate_bert_parameters(config)

# 打印结果
print(f"嵌入层参数: {params['embeddings']:,}")
print(f"Transformer层参数 ({config['num_hidden_layers']}层): {params['transformer_layers']:,}")
print(f"池化层参数: {params['pooler']:,}")
print(f"MLM头部参数: {params['mlm_head']:,}")
print(f"\nBERT基础模型总参数量: {params['bert_base_total']:,} ({params['bert_base_total'] / 1e6:.1f}M)")
print(f"包含MLM头的总参数量: {params['bert_mlm_total']:,} ({params['bert_mlm_total'] / 1e6:.1f}M)")
print(f"模型规模: {params['bert_mlm_total'] / 1e9:.3f}B (0.1B级)")
