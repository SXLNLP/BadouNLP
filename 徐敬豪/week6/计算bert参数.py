"""
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
"return_dict": false,
"num_labels":18
"""

def calculate_bert_parameters(config):
    H = config['hidden_size']
    L = config['num_hidden_layers']
    I = config['intermediate_size']
    V = config['vocab_size']
    P = config['max_position_embeddings']
    T = config['type_vocab_size']

    # 1. 嵌入层参数
    embedding_params = (V + P + T) * H  # 词嵌入 + 位置嵌入 + 句子类型嵌入

    # 2. Transformer层参数（每层）
    # 自注意力层
    attention_params = 4 * (H * H)  # Q,K,V投影矩阵 + 输出投影矩阵

    # 前馈神经网络
    ffn_params = (H * I + I) + (I * H + H)  # 两个线性层（含偏置）

    # 层归一化参数（自注意力和FFN各一个）
    layer_norm_params = 2 * (2 * H)  # 两个层归一化（每个有gamma和beta）

    # 每层总参数
    per_layer_params = attention_params + ffn_params + layer_norm_params

    # 3. 池化层参数
    pooler_params = H * H + H  # 全连接层（含偏置）

    # 总参数
    total_params = embedding_params + (per_layer_params * L) + pooler_params

    return total_params
bert_base_chinese_config = {
    'hidden_size': 768, #隐藏层维度
    'num_hidden_layers': 12, #Transformer层数
    'intermediate_size': 3072, #FFN中间层维度
    'vocab_size': 21128, #词汇表大小
    'max_position_embeddings': 512, #最大位置编码数
    'type_vocab_size': 2, #句子类型数量
    'num_attention_heads': 12  #注意力头数
}
base_params = calculate_bert_parameters(bert_base_chinese_config)
print(f"BERT-base-chinese 总参数: {base_params:,} (约 {base_params/1e6:.1f}M)")
