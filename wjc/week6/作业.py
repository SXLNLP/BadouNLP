hidden_size = 768
vocab_size = 21128
max_position_embeddings = 512
num_attention_heads = 12

count = hidden_size * (vocab_size + max_position_embeddings + 4) + num_attention_heads * (12 * hidden_size * hidden_size + 13 * hidden_size) + hidden_size * hidden_size + hidden_size
print(count)
