import json


def calculate_bert_params(config_dict):
    # 1. embeding层
    token_embedding = config_dict['vocab_size'] * config_dict['hidden_size']  # 词嵌入
    position_embedding = config_dict['max_position_embeddings'] * config_dict['hidden_size']  # 位置嵌入
    segment_embedding = config_dict['type_vocab_size'] * config_dict['hidden_size']  # 段嵌入
    embedding_layer_norm = 2 * config_dict['hidden_size']  # 嵌入层归一化
    embedding_params = (
            token_embedding +
            position_embedding +
            segment_embedding +
            embedding_layer_norm
    )

    # 2. 12个Transformer层
    transformer_params = config_dict['num_hidden_layers'] * transformer_layer_params(config_dict)

    # 3. 池化层
    pooler_layer = config_dict['hidden_size'] * config_dict['hidden_size'] + config_dict['hidden_size']  # 权重矩阵 + 偏置
    total_params = embedding_params + transformer_params + pooler_layer
    return total_params


def transformer_layer_params(config_dict):
    attention_heads = (
            3 * config_dict['hidden_size']
            * (config_dict['hidden_size'] // config_dict['num_attention_heads'])
            * config_dict['num_attention_heads']  # Q/K/V矩阵
    )
    attention_output = config_dict['hidden_size'] * config_dict['hidden_size']  # 注意力输出投影
    attention_bias = config_dict['hidden_size'] * 3 + config_dict['hidden_size']  # Q/K/V偏置 + 输出偏置

    # 全连接
    linear_layer1 = config_dict['hidden_size'] * config_dict['intermediate_size']  # 第一层权重 (H->4H)
    linear_layer2 = config_dict['intermediate_size'] * config_dict['hidden_size']  # 第二层权重 (4H->H)
    linear_bias = config_dict['intermediate_size'] + config_dict['hidden_size']  # 两层偏置

    # normalization层参数
    layer_norm = 2 * config_dict['hidden_size']  # 两个LayerNorm层
    return (
            attention_heads +
            attention_output +
            attention_bias +
            linear_layer1 +
            linear_layer2 +
            linear_bias +
            layer_norm
    )


with open('./config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)
# print(config)
bert_params = calculate_bert_params(config)
print(f"The total num of parameters:: {bert_params:,}")
