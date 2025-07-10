import json

'''
计算bert模型的参数量
'''
def calc_bert_param_size(input_size, hidden_size=768, num_layers=12):
    # 嵌入层
    segment_embedding = 2
    position_embedding = 512
    embedding = input_size * hidden_size + segment_embedding * hidden_size + position_embedding * hidden_size

    # Transformer
    # self-attention
    q_w = hidden_size * hidden_size
    k_w = hidden_size * hidden_size
    v_w = hidden_size * hidden_size
    attention_output_weight = hidden_size * hidden_size
    attn = q_w + k_w + v_w + attention_output_weight
    # Feed Forward
    ffn = hidden_size * (4 * hidden_size) + (4 * hidden_size) * hidden_size
    encoder_layer = attn + ffn

    # 输出层
    pooler = hidden_size * hidden_size

    return embedding + num_layers * encoder_layer + pooler

if __name__ == '__main__':
    vocab = json.load(open('./vocab.json', "r", encoding="utf8"))
    print('词表长度=%d' % len(vocab))
    bert_param_size = calc_bert_param_size(len(vocab), hidden_size=768, num_layers=12)
    print('bert模型参数量=%f' % bert_param_size)

