def embedding_params(vocab_size, max_length, hidden_size):
    token_params = vocab_size * hidden_size
    segment_params = 2 * hidden_size
    position_params = max_length * hidden_size
    layer_norm = hidden_size
    return token_params + segment_params + position_params + layer_norm


def self_attention_params(hidden_size):
    q_params_size = hidden_size * hidden_size + hidden_size
    k_params_size = hidden_size * hidden_size + hidden_size
    v_params_size = hidden_size * hidden_size + hidden_size
    liner_params_size = hidden_size * hidden_size + hidden_size
    return q_params_size + k_params_size + v_params_size + liner_params_size


def feed_forward_params(hidden_size):
    in_params_size = hidden_size * 4 * hidden_size + 4 * hidden_size
    out_params_size = 4 * hidden_size * hidden_size + hidden_size
    return in_params_size + out_params_size


hidden_size = 768
vocab_size = 21128
max_length = 512
layers_num = 12
embedding_params_num = embedding_params(vocab_size, max_length, hidden_size)
print('Embedding层参数数量:', embedding_params_num)
self_attention_params_num = self_attention_params(hidden_size)
print('单层Self-Attention层参数数量:', embedding_params_num)
layer_norm_1 = hidden_size
layer_norm_2 = hidden_size
feed_forward_params_num = feed_forward_params(hidden_size)
print('单层Feed-Forward层参数数量:', feed_forward_params_num)
encoder_params_num = self_attention_params_num + layer_norm_1 + feed_forward_params_num + layer_norm_2
print('单层Encoder层参数数量:', encoder_params_num )
total_params_num = embedding_params_num + layers_num * encoder_params_num
print('总参数数量:', total_params_num)
