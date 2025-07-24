def calculate_bert_params(
    vocab_size=30522,Add commentMore actions
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    use_mlm_head=False,
    reuse_embeddings=True,
):
    """
    计算 BERT 模型的参数量，支持自定义关键参数。
    返回各组件参数量和总参数量。

    参数:
        vocab_size: 词表大小（默认 30522，BERT-base）。
        hidden_size: 隐藏层维度（默认 768）。
        num_layers: Transformer 层数（默认 12）。
        num_attention_heads: 注意力头数（默认 12）。
        intermediate_size: FFN 中间层维度（默认 3072）。
        max_position_embeddings: 最大位置编码长度（默认 512）。
        use_mlm_head: 是否计算 MLM 分类头的参数量（默认 False）。
        reuse_embeddings: 是否复用 Embedding 层的权重（默认 True）。

    返回:
        dict: 包含各组件参数量的字典和总参数量。
    """
    params = {}

    # 1. Embedding 层
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    segment_embeddings = 2 * hidden_size  # 句子A/B
    embedding_layer_norm = 2 * hidden_size  # LayerNorm 的 gamma 和 beta

    params["embedding"] = (
        token_embeddings + position_embeddings + segment_embeddings + embedding_layer_norm
    )

    # 2. 单层 Transformer 的参数
    # (a) 多头注意力
    head_dim = hidden_size // num_attention_heads
    qkv_proj = 3 * hidden_size * head_dim * num_attention_heads  # Q/K/V 矩阵
    output_proj = hidden_size * hidden_size  # 输出投影
    attention_bias = 3 * hidden_size  # Q/K/V 的偏置（可选）

    params["attention_per_layer"] = qkv_proj + output_proj + attention_bias

    # (b) 前馈网络
    ffn_input = hidden_size * intermediate_size
    ffn_output = intermediate_size * hidden_size
    ffn_bias = intermediate_size + hidden_size  # 两层的偏置

    params["feedforward_per_layer"] = ffn_input + ffn_output + ffn_bias

    # (c) LayerNorm（自注意力后和前馈后各一个）
    params["layer_norm_per_layer"] = 2 * 2 * hidden_size  # 两个 LayerNorm

    # 单层 Transformer 总参数量
    params["transformer_per_layer"] = (
        params["attention_per_layer"] + params["feedforward_per_layer"] + params["layer_norm_per_layer"]
    )

    # 所有 Transformer 层
    params["transformer"] = num_layers * params["transformer_per_layer"]

    # 3. 输出层（可选）
    params["mlm_head"] = 0
    if use_mlm_head and not reuse_embeddings:
        params["mlm_head"] = hidden_size * vocab_size  # MLM 分类头

    # 总参数量
    params["total"] = params["embedding"] + params["transformer"] + params["mlm_head"]

    # 转换为百万（M）单位
    for key in params:
        params[key] = params[key] / 1e6

    return params


# 计算 BERT-base 的参数量
bert_base = calculate_bert_params(
    vocab_size=30522,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    reuse_embeddings=True,
    use_mlm_head=False,
)

print("BERT-base 参数量（单位：MB）:")
for key, value in bert_base.items():
    print(f"{key:>20}: {value:.2f}M")
