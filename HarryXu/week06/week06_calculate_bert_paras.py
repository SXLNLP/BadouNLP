vocab_size = 30000  # 词表大小
hidden_size = 768   # 隐藏层维度
max_position_embeddings = 512  # 最大序列长度
type_vocab_size = 2  # 句子分段数 (Segment A/B)
num_hidden_layers = 12  # Transformer层数
intermediate_size = 4 * hidden_size  # 全连接中间层
num_attention_heads = 12  # 注意力头数

def calculate_bert_base_params():
    # 1. embeding层
    token_embedding = vocab_size * hidden_size             # 词嵌入
    position_embedding = max_position_embeddings * hidden_size  # 位置嵌入
    segment_embedding = type_vocab_size * hidden_size     # 段嵌入
    embedding_layer_norm = 2 * hidden_size                # 嵌入层归一化 (gamma+beta)
    
    embedding_params = (
        token_embedding + 
        position_embedding + 
        segment_embedding + 
        embedding_layer_norm
    )
    
    # 2. 12个Transformer层
    transformer_params = num_hidden_layers * transformer_layer_params()
    
    # 3. 池化层
    pooler_layer = hidden_size * hidden_size + hidden_size  # 权重矩阵 + 偏置
    
    total_params = embedding_params + transformer_params + pooler_layer
    return total_params

def transformer_layer_params():
    attention_heads = (
        3 * hidden_size * (hidden_size // num_attention_heads) * num_attention_heads  # Q/K/V矩阵
    )
    attention_output = hidden_size * hidden_size  # 注意力输出投影
    attention_bias = hidden_size * 3 + hidden_size  # Q/K/V偏置 + 输出偏置
        
    # 全连接升维再降维
    ffn_layer1 = hidden_size * intermediate_size   # 第一层权重 (H->4H)
    ffn_layer2 = intermediate_size * hidden_size   # 第二层权重 (4H->H)
    ffn_bias = intermediate_size + hidden_size     # 两层偏置
        
    # normalization层参数
    layer_norm = 2 * hidden_size  # 两个LayerNorm层
    return (
        attention_heads + 
        attention_output + 
        attention_bias + 
        ffn_layer1 + 
        ffn_layer2 + 
        ffn_bias + 
        layer_norm
    )

bert_base_params = calculate_bert_base_params()
print(f"Totol num of parameters:: {bert_base_params:,}")
