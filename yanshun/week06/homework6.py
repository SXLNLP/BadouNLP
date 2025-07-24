import torch
from transformers import BertModel

import json
config_path = "./bert-base-chinese/config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
vocab_size = config['vocab_size']
max_position_embeddings = config['max_position_embeddings']
type_vocab_size = config['type_vocab_size']
hidden_size = config['hidden_size']
intermediate_size = config['intermediate_size']
num_attention_heads = config['num_attention_heads']
num_hidden_layers = config['num_hidden_layers']
pooler_fc_size = hidden_size

def calc_embedding_params():
    word_emb = vocab_size * hidden_size
    position_emb = max_position_embeddings * hidden_size
    token_type_emb = type_vocab_size * hidden_size
    layer_norm = 2 * hidden_size
    return word_emb + position_emb + token_type_emb + layer_norm

def calc_self_attention_params():
    q_proj = hidden_size * hidden_size + hidden_size
    k_proj = hidden_size * hidden_size + hidden_size
    v_proj = hidden_size * hidden_size + hidden_size
    output_proj = hidden_size * hidden_size + hidden_size
    attention_output_ln = 2 * hidden_size
    return q_proj + k_proj + v_proj + output_proj + attention_output_ln

def calc_feed_forward_params():
    intermediate_proj = hidden_size * intermediate_size + intermediate_size
    output_proj = intermediate_size * hidden_size + hidden_size
    output_ln = 2 * hidden_size
    return intermediate_proj + output_proj + output_ln

def calc_transformer_layer_params():
    return calc_self_attention_params() + calc_feed_forward_params()

def calc_pooler_params():
    return hidden_size * pooler_fc_size + pooler_fc_size

embedding_params = calc_embedding_params()
per_layer_params = calc_transformer_layer_params()
total_transformer_params = per_layer_params * num_hidden_layers
pooler_params = calc_pooler_params()
total_params = embedding_params + total_transformer_params + pooler_params

print(f"Embedding层参数: {embedding_params:,}")
print(f"单Transformer层参数: {per_layer_params:,}")
print(f"总Transformer参数: {total_transformer_params:,}")
print(f"Pooler层参数: {pooler_params:,}")
print(f"BERT总参数量: {total_params:,}")
"""
Embedding层参数: 16,622,592
单Transformer层参数: 7,087,872
总Transformer参数: 85,054,464
Pooler层参数: 590,592
BERT总参数量: 102,267,648
"""
print()

bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
bert.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def detailed_parameter_count(model):
    params = {}
    embedding_params = 0
    embedding_params += model.embeddings.word_embeddings.weight.numel()
    embedding_params += model.embeddings.position_embeddings.weight.numel()
    embedding_params += model.embeddings.token_type_embeddings.weight.numel()
    embedding_params += model.embeddings.LayerNorm.weight.numel()
    embedding_params += model.embeddings.LayerNorm.bias.numel()
    params['embeddings'] = embedding_params
    encoder_params = 0
    for layer in model.encoder.layer:
        layer_params = 0
        layer_params += layer.attention.self.query.weight.numel()
        layer_params += layer.attention.self.query.bias.numel()
        layer_params += layer.attention.self.key.weight.numel()
        layer_params += layer.attention.self.key.bias.numel()
        layer_params += layer.attention.self.value.weight.numel()
        layer_params += layer.attention.self.value.bias.numel()
        layer_params += layer.attention.output.dense.weight.numel()
        layer_params += layer.attention.output.dense.bias.numel()
        layer_params += layer.attention.output.LayerNorm.weight.numel()
        layer_params += layer.attention.output.LayerNorm.bias.numel()
        layer_params += layer.intermediate.dense.weight.numel()
        layer_params += layer.intermediate.dense.bias.numel()
        layer_params += layer.output.dense.weight.numel()
        layer_params += layer.output.dense.bias.numel()
        layer_params += layer.output.LayerNorm.weight.numel()
        layer_params += layer.output.LayerNorm.bias.numel()
        encoder_params += layer_params
    layer_params = encoder_params / len(model.encoder.layer)
    params['encoder_layer_avg'] = int(layer_params)
    params['encoder'] = encoder_params
    pooler_params = 0
    pooler_params += model.pooler.dense.weight.numel()
    pooler_params += model.pooler.dense.bias.numel()
    params['pooler'] = pooler_params
    return params
detailed_params = detailed_parameter_count(bert)
total_params = count_parameters(bert)
print(f"BERT 模型总参数量: {total_params:,}")
print(f"Embedding 层: {detailed_params['embeddings']:,}")
print(f"Transformer 编码器 (总): {detailed_params['encoder']:,}")
print(f"单个 Transformer 层: {detailed_params['encoder_layer_avg']:,}")
print(f"Pooler 层: {detailed_params['pooler']:,}")
"""
BERT 模型总参数量: 102,267,648
Embedding 层: 16,622,592
Transformer 编码器 (总): 85,054,464
单个 Transformer 层: 7,087,872
Pooler 层: 590,592
"""
