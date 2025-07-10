import json

'''
计算bert参数量

'''
config = json.load(open(r"config.json"))
hidden_size = config["hidden_size"]
vocab_size = config["vocab_size"]
type_vocab_size = config["type_vocab_size"]
max_position_embeddings = config["max_position_embeddings"]
num_attention_heads = config["num_attention_heads"]
num_hidden_layers = config["num_hidden_layers"]
intermediate_size = config["intermediate_size"]

embedding_size = hidden_size * vocab_size + hidden_size * type_vocab_size + hidden_size * max_position_embeddings

dk = hidden_size // num_attention_heads
self_attention_size = (hidden_size * 3 * dk * num_attention_heads + hidden_size * dk * num_attention_heads) * num_hidden_layers

feed_forward_size = (hidden_size * intermediate_size + hidden_size * intermediate_size) * num_hidden_layers

print(embedding_size + self_attention_size + feed_forward_size)
