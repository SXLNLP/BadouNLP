"""
计算bert参数量
embedding层、transformer（self-attention、feed-forward）、pooler层
最后算出来结果：95134464，大概380M
"""
import json

# Load config file
config_path = "/Users/ge/PycharmProjects/pythonProject/bd/week06/bert-base-chinese/config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Extract config values
vocab_size = config['vocab_size']
max_position_embeddings = config['max_position_embeddings']
type_vocab_size = config['type_vocab_size']
hidden_size = config['hidden_size']
intermediate_size = config['intermediate_size']
num_attention_heads = config['num_attention_heads']
num_hidden_layers = config['num_hidden_layers']
pooler_fc_size = config['pooler_fc_size']

# Embedding layer parameters
embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size + type_vocab_size * hidden_size + 2 * hidden_size

# Self-attention parameters per layer
self_attention_params = 3 * hidden_size * hidden_size + 2 * hidden_size

# Feed-forward parameters per layer
feed_forward_params = hidden_size * intermediate_size + intermediate_size + intermediate_size * hidden_size + hidden_size

# Transformer layer parameters
transformer_layer_params = self_attention_params + feed_forward_params

# Total transformer parameters
total_transformer_params = transformer_layer_params * num_hidden_layers

# Pooler layer parameters
pooler_params = hidden_size * pooler_fc_size + pooler_fc_size

# Total parameters
total_params = embedding_params + total_transformer_params + pooler_params

print(f"Total parameters: {total_params}")
