def calculate_bert_params(
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        pooler_size=768,

):
    """
    手动计算BERT模型的参数量

    Args:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        num_hidden_layers: Transformer层数
        num_attention_heads: 注意力头数
        intermediate_size: 前馈网络中间层维度
        max_position_embeddings: 最大位置编码长度
        type_vocab_size: 段类型数
        pooler_size: 池化层维度


    Returns:
        参数量字典，包含各组件参数量和总参数量
    """
    params = {}

    # 1. 嵌入层参数量
    embedding_params = {}

    # 词嵌入
    embedding_params["word_embedding"] = vocab_size * hidden_size

    # 位置嵌入
    embedding_params["position_embedding"] = max_position_embeddings * hidden_size

    # 段嵌入
    embedding_params["token_type_embedding"] = type_vocab_size * hidden_size

    # 层归一化
    embedding_params["layer_norm"] = 2 * hidden_size  # gamma和beta参数

    # 嵌入层总参数量
    params["embedding"] = sum(embedding_params.values())

    # 2. 单个Transformer层参数量
    layer_params = {}

    # 多头注意力参数量
    attention_params = {}
    # QKV投影矩阵
    attention_params["qkv_weights"] = 3 * hidden_size * hidden_size
    # QKV偏置
    attention_params["qkv_biases"] = 3 * hidden_size
    # 输出投影矩阵
    attention_params["output_weights"] = hidden_size * hidden_size
    # 输出偏置
    attention_params["output_biases"] = hidden_size
    # 注意力总参数量
    layer_params["attention"] = sum(attention_params.values())

    # 前馈网络参数量
    ff_params = {}
    # 第一层线性变换
    ff_params["intermediate_weights"] = hidden_size * intermediate_size
    ff_params["intermediate_biases"] = intermediate_size
    # 第二层线性变换
    ff_params["output_weights"] = intermediate_size * hidden_size
    ff_params["output_biases"] = hidden_size
    # 前馈网络总参数量
    layer_params["feed_forward"] = sum(ff_params.values())

    # 层归一化参数量
    layer_params["layer_norm"] = 2 * 2 * hidden_size  # 两个层归一化层

    # 单个Transformer层总参数量
    params["per_layer"] = sum(layer_params.values())

    # 所有Transformer层参数量
    params["all_layers"] = params["per_layer"] * num_hidden_layers

    # 3. 输出层参数量
    output_params = {}

    # 池化层参数量
    output_params["pooler"] = hidden_size * pooler_size + pooler_size

    # 输出层总参数量
    params["output"] = sum(output_params.values())

    # 4. 总参数量
    params["total"] = params["embedding"] + params["all_layers"] + params["output"]

    return params


# 计算BERT-Base参数量
bert_base_params = calculate_bert_params(
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2
)


# 打印结果
def print_params(model_name, params):
    print(f"\n{model_name}参数量统计:")
    print(f"  嵌入层: {params['embedding']:,}")
    print(f"  每层Transformer: {params['per_layer']:,}")
    print(f"  所有Transformer层: {params['all_layers']:,}")
    print(f"  输出层: {params['output']:,}")
    print(f"  总计: {params['total']:,}")


print_params("BERT-Base", bert_base_params)
