def calculate_bert_params(config):
    """
    计算BERT模型参数量（基于配置文件）

    参数说明:
    - config: BERT配置文件字典
    """
    # 从配置文件中提取关键参数
    V = config["vocab_size"]  # 词表大小 (21128)
    H = config["hidden_size"]  # 隐藏层维度 (768)
    L = config["num_hidden_layers"]  # Transformer层数 (12)
    I = config["intermediate_size"]  # FFN中间层维度 (3072)
    P = config["max_position_embeddings"]  # 最大位置数 (512)
    T = config["type_vocab_size"]  # 句子类型数 (2)

    total_params = 0

    # ==================== 1. Embedding层参数 ====================
    # 词嵌入矩阵 (V × H)
    word_embedding_params = V * H
    # 位置嵌入矩阵 (P × H)
    position_embedding_params = P * H
    # 句子类型嵌入矩阵 (T × H)
    token_type_embedding_params = T * H
    # Embedding层归一化参数 (gamma + beta)
    embedding_ln_params = 2 * H

    embedding_total = (word_embedding_params + position_embedding_params +
                       token_type_embedding_params + embedding_ln_params)
    total_params += embedding_total

    # ==================== 2. Transformer层参数 ====================
    per_layer_params = 0

    # 注意力层参数 (Q/K/V矩阵)
    # 每个Q/K/V矩阵: H × (H/12) × 12头 = H × H
    attention_qkv_params = 3 * (H * H + H)  # 权重矩阵 + 偏置

    # 注意力输出层 (H × H)
    attention_output_params = H * H + H

    # 注意力层归一化 (gamma + beta)
    attention_ln_params = 2 * H

    # FFN第一层 (H × I)
    ffn_intermediate_params = H * I + I

    # FFN第二层 (I × H)
    ffn_output_params = I * H + H

    # FFN层归一化 (gamma + beta)
    ffn_ln_params = 2 * H

    # 单层Transformer总参数
    per_layer_params = (attention_qkv_params + attention_output_params +
                        attention_ln_params + ffn_intermediate_params +
                        ffn_output_params + ffn_ln_params)

    # 所有Transformer层参数
    transformer_total = L * per_layer_params
    total_params += transformer_total

    # ==================== 3. Pooling层参数 ====================
    # 标准BERT只包含一个全连接层 (H × H)
    pooling_params = H * H + H  # 权重矩阵 + 偏置
    total_params += pooling_params

    # ==================== 结果输出 ====================
    print("=" * 60)
    print(f"BERT模型参数量计算 (基于配置文件)")
    print("=" * 60)
    print(f"词表大小 (V): {V}")
    print(f"隐藏层维度 (H): {H}")
    print(f"Transformer层数 (L): {L}")
    print(f"最大位置编码数 (P): {P}")
    print(f"句子类型数 (T): {T}")
    print("-" * 60)
    print(f"Embedding层参数: {embedding_total:,} (占比 {embedding_total / total_params:.1%})")
    print(f"  ├─ 词嵌入: {word_embedding_params:,}")
    print(f"  ├─ 位置嵌入: {position_embedding_params:,}")
    print(f"  ├─ 句子类型嵌入: {token_type_embedding_params:,}")
    print(f"  └─ 归一化参数: {embedding_ln_params:,}")
    print(f"Transformer层参数: {transformer_total:,} (占比 {transformer_total / total_params:.1%})")
    print(f"  ├─ 单层参数: {per_layer_params:,}")
    print(f"  └─ 总参数: {transformer_total:,} ({L}层)")
    print(f"Pooling层参数: {pooling_params:,}")
    print("-" * 60)
    print(f"总参数量: {total_params:,} (约 {total_params / 1e6:.1f}M)")
    print("=" * 60)

    return total_params


# 使用示例
bert_config = {
      "architectures": [
        "BertForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.1,
      "directionality": "bidi",
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "pooler_fc_size": 768,
      "pooler_num_attention_heads": 12,
      "pooler_num_fc_layers": 3,
      "pooler_size_per_head": 128,
      "pooler_type": "first_token_transform",
      "type_vocab_size": 2,
      "vocab_size": 21128,
      "return_dict": False,
      "num_labels":18
    }

total_params = calculate_bert_params(bert_config)
