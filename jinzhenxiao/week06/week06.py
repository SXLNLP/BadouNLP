#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT模型参数数量计算脚本
计算并输出bert模型总参数量
"""


def calculate_bert_parameters(vocab_size=30522, hidden_size=768, num_layers=12, intermediate_size=3072,
                              max_position_embeddings=512, type_vocab_size=2):
    # 1. 嵌入层参数
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    token_type_embeddings = type_vocab_size * hidden_size
    embedding_layernorm = 2 * hidden_size
    embedding_params = token_embeddings + position_embeddings + token_type_embeddings + embedding_layernorm

    # 2. 单层Transformer参数
    attention_qkv = 3 * hidden_size * hidden_size + 3 * hidden_size  # Q,K,V权重和偏置
    attention_output = hidden_size * hidden_size + hidden_size  # 输出投影
    attention_layernorm = 2 * hidden_size
    single_attention = attention_qkv + attention_output + attention_layernorm

    # 前馈网络
    ffn_dense1 = hidden_size * intermediate_size + intermediate_size
    ffn_dense2 = intermediate_size * hidden_size + hidden_size
    ffn_layernorm = 2 * hidden_size
    single_ffn = ffn_dense1 + ffn_dense2 + ffn_layernorm

    # 3. 所有Transformer层参数
    transformer_params = num_layers * (single_attention + single_ffn)

    # 4. 池化层参数
    pooler_params = hidden_size * hidden_size + hidden_size

    # 5. 总参数
    return embedding_params + transformer_params + pooler_params


def main():
    bert_base_params = calculate_bert_parameters()
    print(f"BERT-Base总参数量: {bert_base_params}")


if __name__ == "__main__":
    main()