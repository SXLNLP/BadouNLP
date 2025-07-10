from transformers import BertConfig, BertModel
import numpy as np

bert_config = BertConfig.from_pretrained(r"E:\AI_study\八斗学院\录播课件\第六周\bert-base-chinese", return_dict=False)
bert = BertModel.from_pretrained(r"E:\AI_study\八斗学院\录播课件\第六周\bert-base-chinese", return_dict=False)

param_count = 0 # 总参数量
hidden_size = bert_config.hidden_size  # 字向量化维度
vocab_size = bert_config.vocab_size  # 字表总数
type_vocab_size = bert_config.type_vocab_size  # 句子类型表大小
max_position_embeddings = bert_config.max_position_embeddings  # 最大句子长度
intermediate_size = bert_config.intermediate_size  # 隐藏层扩充后的维度 768 * 4 == 3072

"""
embedding 层
"""
# 字表向量化
word2vector_embeddings = hidden_size * vocab_size
# 句子间 token，区分两个句子
token2sentence_embeddings = hidden_size * type_vocab_size
# 单字位置顺序的向量参数，最大 512
max_positions_embeddings = hidden_size * max_position_embeddings
# 三层 embedding 加和之后，进行一次归一化操作
# LayerNorm.weight：是一个长度为 768 的向量，表示对归一化后的 768 个特征分别乘以不同的系数（放大或缩小）。
# LayerNorm.bias：是一个长度为 768 的向量，表示对归一化后的 768 个特征分别加上不同的偏移值。
norm_weight_embeddings = hidden_size
norm_bias_embeddings = hidden_size

param_count  = np.sum([param_count, word2vector_embeddings, token2sentence_embeddings, max_positions_embeddings, norm_weight_embeddings, norm_bias_embeddings])


'''
使用一层 transformer，或者将transformer 所有调用的权重共享
多头自注意力机制 默认 12个
'''
# layer_query  _query
query_weight_layer = hidden_size * hidden_size
query_bias_layer = hidden_size
# layer_key _key
key_weight_layer = hidden_size * hidden_size
key_bias_layer = hidden_size
# layer_value _value
value_weight_layer = hidden_size * hidden_size
value_bias_layer = hidden_size
# QKV_output -- 线性层
output_QKV_weight_layer = hidden_size * hidden_size
output_QKV_bias_layer = hidden_size
# 归一化 -- 线性层 embedding层 + self_attention层
output_QKV_weight_layerNorm = hidden_size
output_QKV_bias_layerNorm = hidden_size

single_transformer_layer_qkv_params = np.sum([query_weight_layer, query_bias_layer, key_weight_layer, key_bias_layer, value_weight_layer, value_bias_layer, output_QKV_weight_layer, output_QKV_bias_layer, output_QKV_weight_layerNorm, output_QKV_bias_layerNorm])

'''
中间层全连接层 —— 维度扩充 4 倍 —— # bert_config.intermediate_size —— 768*4=3072
激活函数 gelu
'''
intermediate_gelu_weight_layer = hidden_size * intermediate_size
intermediate_gelu_bias_layer = intermediate_size
# 线性层
intermediate_linear_weight_layer = intermediate_size * hidden_size
intermediate_linear_bias_layer = hidden_size
# 归一化线性层
output_linear_weight_layerNorm = hidden_size
output_linear_bias_layerNorm = hidden_size

single_transformer_layer_intermediate_params = np.sum([intermediate_gelu_weight_layer, intermediate_gelu_bias_layer, intermediate_linear_weight_layer, intermediate_linear_bias_layer, output_linear_weight_layerNorm, output_linear_bias_layerNorm])

param_count = np.sum([param_count, single_transformer_layer_qkv_params, single_transformer_layer_intermediate_params])
print(f'单层transformer 总参数：{single_transformer_layer_qkv_params + single_transformer_layer_intermediate_params}')

'''
对 transformer 最后输出的隐藏层进行池化操作 （全连接层 + tanh激活函数）
'''
pool_weight = hidden_size * hidden_size
pool_bias = hidden_size
param_count = np.sum([param_count, pool_weight, pool_bias])


print("bert 1层transformer 总参数：", param_count)

'''
遍历 bert.state_dict()，计算参数量
'''
parameters_amount = 0
for key, value in bert.state_dict().items():
    # print(f'key-value.shape: {key} —— {value.shape}')
    temp_size = 1
    for size in value.shape:
        temp_size *= size
    parameters_amount += temp_size
print(f'bert 2层transformer 总参数: {parameters_amount}')

print(f'双层transformer 总参数比 单层 transformer总参数多：{parameters_amount - param_count}')