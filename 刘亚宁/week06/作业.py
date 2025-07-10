vocab_size = 21128
max_position_embeddings = 512
type_vocab_size = 2
hidden_size = 768
num_attention_heads = 12
intermediate_size = 3072

def calculate_embedding():
	calculate_embedding_count = (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size
	return calculate_embedding_count
	
def self_attention():
	self_attention_count = 4 * hidden_size * (hidden_size / num_attention_heads) * num_attention_heads * 12
	return self_attention_count
	
def feed_forward():
	feed_forward_count = 2 * hidden_size * intermediate_size * 12
	return feed_forward_count
	
def calculate_layerNorm():
	layerNorm_count = 2 * hidden_size * 12
	return layerNorm_count
	
def calculate_pool():
	pool_count = hidden_size * hidden_size
	return pool_count
	
total_count = calculate_embedding() + self_attention() + feed_forward() + calculate_layerNorm() + calculate_pool()
