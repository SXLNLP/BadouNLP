embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

# pool_fc层参数
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
all_paramerters = embedding_parameters + (self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters) * num_layers + pool_fc_parameters
print("计算参数个数为%d" % all_paramerters)
