# -*- coding: utf-8 -*-
"""
@Time ： 2025/6/13 17:16
@Auth ： fengbangwei
@File ：homework_kmeans.py

"""
from transformers import BertModel
import numpy as np
import torch


def load_weights(state_dict, num_layers):
    # embedding部分
    word_embeddings_rows, word_embeddings_cols = state_dict[
        "embeddings.word_embeddings.weight"].numpy().shape  # (21128, 768)
    word_embeddings = word_embeddings_rows * word_embeddings_rows
    position_embeddings_rows, position_embeddings_cols = state_dict[
        "embeddings.position_embeddings.weight"].numpy().shape  # (512, 768)
    position_embeddings = position_embeddings_rows * position_embeddings_cols
    token_type_embeddings_rows, token_type_embeddings_cols = state_dict[
        "embeddings.token_type_embeddings.weight"].numpy().shape  # (2,768)
    token_type_embeddings = token_type_embeddings_rows * token_type_embeddings_cols
    embeddings_layer_norm_weight_rows = state_dict["embeddings.LayerNorm.weight"].numpy().shape[0]  # (768,)
    embeddings_layer_norm_bias_rows = state_dict["embeddings.LayerNorm.bias"].numpy().shape[0]  # (768,)
    embedding = word_embeddings + position_embeddings + token_type_embeddings + embeddings_layer_norm_weight_rows + embeddings_layer_norm_bias_rows
    # transformer部分，有多层
    q_w_rows, q_w_cols = state_dict["encoder.layer.%d.attention.self.query.weight" % 0].numpy().shape  # (768, 768)
    q_w = q_w_rows * q_w_cols
    q_b_rows = state_dict["encoder.layer.%d.attention.self.query.bias" % 0].numpy().shape[0]  # (768,)
    k_w_rows, k_w_cols = state_dict["encoder.layer.%d.attention.self.key.weight" % 0].numpy().shape  # (768, 768)
    k_w = k_w_rows * k_w_cols
    k_b_rows = state_dict["encoder.layer.%d.attention.self.key.bias" % 0].numpy().shape[0]  # (768,)
    v_w_rows, v_w_cols = state_dict["encoder.layer.%d.attention.self.value.weight" % 0].numpy().shape  # (768, 768)
    v_w = v_w_rows * v_w_cols
    v_b_rows = state_dict["encoder.layer.%d.attention.self.value.bias" % 0].numpy().shape[0]  # (768,)
    attention_output_weight_rows, attention_output_weight_cols = state_dict[
        "encoder.layer.%d.attention.output.dense.weight" % 0].numpy().shape  # (768, 768)
    attention_output_weight = attention_output_weight_rows * attention_output_weight_cols
    attention_output_bias_rows = state_dict["encoder.layer.%d.attention.output.dense.bias" % 0].numpy().shape[
        0]  # (768,)
    attention_layer_norm_w_rows = state_dict[
        "encoder.layer.%d.attention.output.LayerNorm.weight" % 0].numpy().shape[0]  # (768,)
    attention_layer_norm_b_rows = state_dict[
        "encoder.layer.%d.attention.output.LayerNorm.bias" % 0].numpy().shape[0]  # (768,)
    intermediate_weight_rows, intermediate_weight_cols = state_dict[
        "encoder.layer.%d.intermediate.dense.weight" % 0].numpy().shape  # (3072, 768)
    intermediate_weight = intermediate_weight_rows * intermediate_weight_cols
    intermediate_bias_rows = state_dict["encoder.layer.%d.intermediate.dense.bias" % 0].numpy().shape[0]  # (3072,)
    output_weight_rows, output_weight_cols = state_dict[
        "encoder.layer.%d.output.dense.weight" % 0].numpy().shape  # (768, 3072)
    output_weight = output_weight_rows * output_weight_cols
    output_bias_rows = state_dict["encoder.layer.%d.output.dense.bias" % 0].numpy().shape[0]  # (768,)
    ff_layer_norm_w_rows = state_dict["encoder.layer.%d.output.LayerNorm.weight" % 0].numpy().shape[0]  # (768,)
    ff_layer_norm_b_rows = state_dict["encoder.layer.%d.output.LayerNorm.bias" % 0].numpy().shape[0]  # (768,)
    transformer = num_layers * (
            q_w + q_b_rows + k_w + k_b_rows + v_w + v_b_rows + attention_output_weight + attention_output_bias_rows
            + attention_layer_norm_w_rows + attention_layer_norm_b_rows + intermediate_weight + intermediate_bias_rows
            + output_weight + output_bias_rows + ff_layer_norm_w_rows + ff_layer_norm_b_rows)

    # pooler层
    pooler_dense_weight_rows, pooler_dense_bias_cols = state_dict["pooler.dense.weight"].numpy().shape  # (768, 768)
    pooler_dense_weight = pooler_dense_weight_rows * pooler_dense_bias_cols
    pooler_dense_bias_rows = state_dict["pooler.dense.bias"].numpy().shape[0]  # (768,)
    pooler = pooler_dense_weight + pooler_dense_bias_rows
    print("embedding层参数量:", embedding)
    print("transformer层参数量:", transformer)
    print("pooler层参数量:", pooler)
    return embedding + transformer + pooler


if __name__ == '__main__':
    bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\AI\nlp\第六周 语言模型\bert-base-chinese",
                                     return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()
    x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
    # torch_x = torch.LongTensor([x])  # pytorch形式输入
    torch_x = torch.from_numpy(np.array([x]))
    seqence_output, pooler_output = bert(torch_x)
    print("bert层总的参数量:", load_weights(state_dict, 1))
    # print(seqence_output.shape, pooler_output.shape)
    # print(seqence_output, pooler_output)
    # print(bert.state_dict().keys())  # 查看所有的权值矩阵名称
