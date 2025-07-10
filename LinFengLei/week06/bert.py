def calculate_bert_params():
    # BERT-base 超参数
    vocab_size = 30522
    hidden_size = 768
    max_position_embeddings = 512
    type_vocab_size = 2
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 4 * hidden_size  # 3072
    
    total_params = 0
    components = {}

    # 1. 嵌入层 (Token, Position, Segment)
    embedding_params = (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size
    components["嵌入层 (Embeddings)"] = embedding_params
    total_params += embedding_params

    # 2. 嵌入层后的层归一化 (LayerNorm)
    embedding_layer_norm = 2 * hidden_size  # gamma 和 beta
    components["嵌入层归一化 (Embedding LayerNorm)"] = embedding_layer_norm
    total_params += embedding_layer_norm

    # 3. Transformer 层 (每层)
    per_layer_params = 0
    
    # 自注意力机制
    # Q, K, V 投影矩阵 (每个: hidden_size x (hidden_size/num_heads))
    # 实际实现中通常使用一个大的权重矩阵然后分割
    self_attn_weight = hidden_size * hidden_size * 3  # W_q, W_k, W_v
    self_attn_bias = hidden_size * 3  # 三个偏置向量
    
    # 自注意力输出矩阵
    self_attn_output_weight = hidden_size * hidden_size
    self_attn_output_bias = hidden_size
    
    # 自注意力总计
    self_attn_total = self_attn_weight + self_attn_bias + self_attn_output_weight + self_attn_output_bias
    
    # 前馈神经网络
    ff_intermediate_weight = hidden_size * intermediate_size
    ff_intermediate_bias = intermediate_size
    ff_output_weight = intermediate_size * hidden_size
    ff_output_bias = hidden_size
    ff_total = ff_intermediate_weight + ff_intermediate_bias + ff_output_weight + ff_output_bias
    
    # 层归一化 (两个: 自注意力后和前馈后)
    layer_norm_params = 2 * (2 * hidden_size)  # 每个层归一化有gamma和beta
    
    # 每层总计
    per_layer_params = self_attn_total + ff_total + layer_norm_params
    components["单Transformer层"] = per_layer_params
    
    # 所有Transformer层
    transformer_params = per_layer_params * num_hidden_layers
    components["Transformer层总计"] = transformer_params
    total_params += transformer_params

    # 输出结果
    print("="*50)
    print(f"BERT-base 参数量计算 (总参数量: {total_params:,})")
    print("="*50)
    print("{:<35} {:>15} {:>15}".format("组件", "参数量", "累计占比"))
    print("-"*65)
    
    cumulative = 0
    for name, params in components.items():
        cumulative += params
        print("{:<35} {:>15,} {:>14.1%}".format(
            name, 
            params,
            cumulative/total_params
        ))
    
    print("="*65)
    print(f"总参数量: {total_params:,} ≈ {total_params/1e6:.1f} million")
    print("注意: 实际实现中通常报告为110M，差异可能来自:")
    print("  - 实现细节(如偏置项的处理)")
    print("  - 未包含的池化层/任务特定层")
    print("  - 嵌入层实现方式差异")
    
    return total_params

# 执行计算
if __name__ == "__main__":
    calculate_bert_params()
