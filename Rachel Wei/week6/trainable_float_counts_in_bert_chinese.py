# Load model directly from Hugging Face
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoConfig

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
config = AutoConfig.from_pretrained("google-bert/bert-base-chinese")

state_dict = model.state_dict()


## Read Config and get the figures 
print(config)
vocab_size = 21128
hidden_size = 768
max_position_embeddings = 512
num_attention_heads = 12
num_hidden_layers = 12
type_vocab_size = 2
pooler_fc_size = 768
pooler_num_attention_heads = 12
pooler_num_fc_layers = 3
pooler_size_per_head = 128

## Embedding layer
token_embedding = vocab_size * hidden_size
segment_embedding = 2 * hidden_size
position_embedding = max_position_embeddings * hidden_size
total_embedding = token_embedding + segment_embedding + position_embedding
print("total_embedding layer has ", total_embedding / 1000000, " trainable floats")

## Self Attention Layer
query_fc_weight = hidden_size * hidden_size
query_fc_bias = hidden_size
key_fc_weight = hidden_size * hidden_size
key_fc_bias = hidden_size
value_fc_weight = hidden_size * hidden_size
value_fc_bias = hidden_size 

self_attention_linear_weight = hidden_size * hidden_size
self_attention_linear_bias = hidden_size

self_attention_layer_norm_weight = hidden_size 
self_attention_layer_norm_bias = hidden_size 

single_layer_self_attention_floats = query_fc_weight + query_fc_bias \
                                   + key_fc_weight + key_fc_bias \
                                   + value_fc_weight + value_fc_bias \
                                   + self_attention_linear_weight + self_attention_linear_bias \
                                   + self_attention_layer_norm_weight + self_attention_layer_norm_bias
print('total_self_attention layer has ', single_layer_self_attention_floats / 1000000, " trainable floats")

## Feed Forward
feed_forward_linear_1_weights = (hidden_size * 4) * hidden_size
feed_forward_linear_1_bias = (hidden_size * 4)
feed_forward_linear_2_weights = hidden_size * (hidden_size * 4)
feed_forward_linear_2_bias = hidden_size
feed_forward_layer_norm_weights = hidden_size
feed_forward_layer_norm_bias = hidden_size

single_layer_feed_forward_floats = feed_forward_linear_1_weights + feed_forward_linear_1_bias + feed_forward_linear_2_weights + feed_forward_linear_2_bias + feed_forward_layer_norm_weights + feed_forward_layer_norm_bias
print("total_feed_forward layer has ", single_layer_feed_forward_floats / 1000000, " trainable floats")

## Model Total
total_multi_layer = num_hidden_layers * (single_layer_self_attention_floats + single_layer_feed_forward_floats)
total_model = total_embedding + total_multi_layer
print("all layers of the self-attention and feed forward has ", total_model / 1000000, " trainable floats")


## Compare with the state_dict figures
total_params = sum(p.numel() for p in state_dict.values())
print(f"The model's state_dict has {total_params / 1000000} trainable floats")
print(f"Difference between my calculation and the actual number of weights is {total_params / 1000000 - total_model / 1000000}. The difference comes from the cls.predictions.decoder.weight, which shared model.bert.embeddings.word_embeddings.weight")


## Details of the state_dict floats
for name, param in model.named_parameters():
    print(f"{name:60} {param.shape} => {param.numel()}")
