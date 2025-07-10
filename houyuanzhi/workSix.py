from transformers import BertModel

def calculate_bert_parameters(
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    share_embedding_weights=True
):
    """
    计算 BERT 模型参数量
    :param vocab_size: 中文Bert词汇表大小
    :param hidden_size: 隐藏层维度
    :param num_hidden_layers: Transformer 层数
    :param intermediate_size: FFN 中间层维度
    :param max_position_embeddings: 最大序列长度
    :param share_embedding_weights: 是否共享 Embedding 和输出层权重
    :return: 总参数量
    """
    
    # 1. Embedding 层参数量
    token_embeddings = vocab_size * hidden_size
    segment_position_embeddings = 2 * max_position_embeddings * hidden_size
    embedding_total = token_embeddings + segment_position_embeddings
    
    # 2. Transformer 编码层参数量
    # 自注意力机制 (Q/K/V 投影 + 输出投影)
    attention_weights = 3 * (hidden_size * hidden_size)  # Q/K/V 权重矩阵
    attention_bias = 3 * hidden_size                    # Q/K/V 偏置项
    attention_output = hidden_size * hidden_size + hidden_size  # 输出投影层
    
    # 前馈网络 (FFN)
    ffn_weights = 2 * (hidden_size * intermediate_size)  # 两层线性变换
    ffn_bias = 2 * intermediate_size + hidden_size       # 偏置项 + LayerNorm
    
    # 单层 Transformer 参数量
    transformer_layer = attention_weights + attention_bias + attention_output + ffn_weights + ffn_bias
    
    # 所有 Transformer 层总参数量
    transformer_total = transformer_layer * num_hidden_layers
    
    # 3. 输出层参数量
    output_layer = vocab_size * hidden_size if not share_embedding_weights else 0
    
    # 4. 总参数量
    total_params = embedding_total + transformer_total + output_layer
    
    # 打印详细参数分布
    print("参数分布分析:")
    print(f"1. Embedding 层: {embedding_total:,} ")
    print(f"   - Token Embeddings: {token_embeddings:,}")
    print(f"   - Segment/Position Embeddings: {segment_position_embeddings:,}")
    print(f"2. Transformer 编码层: {transformer_total:,} ")
    print(f"   - 单层参数量: {transformer_layer:,}")
    print(f"   - 层数: {num_hidden_layers}")
    print(f"3. 输出层: {output_layer:,} ")
    print(f"\n 总参数量: {total_params:,} ")
    model = BertModel.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False)
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"模型直接获取到的参数量：{model_total_params:,}")

    return total_params

# 计算 BERT-Base 参数量
calculate_bert_parameters()


