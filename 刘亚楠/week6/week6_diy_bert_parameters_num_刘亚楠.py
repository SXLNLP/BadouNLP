
"""

手动计算bert-chinese 模型中的总参数个数
"""

from transformers import BertModel
bert = BertModel.from_pretrained(r"/Users/nlp/week6/第六周 语言模型/bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
#print(word_embeddings.shape[0],"----词表大小")


vocab_length = word_embeddings.shape[0] # 词表大小
max_len = 512 # 最大序列长度
sen_class = 2 # 句子类型数
hidden_size = 768 # 隐藏层维度

# 三个embdding层参数个数
word_embeddings_num = vocab_length * hidden_size
position_embeddings_num = max_len * hidden_size
token_type_embeddings_num = sen_class * hidden_size
# norm_layer层参数个数
norm_layer_embeddings_num = hidden_size * 2 # y=wx+b 因为有w和b所以乘2
# 四者相加
diy_embeddings_num = word_embeddings_num+position_embeddings_num+token_type_embeddings_num+norm_layer_embeddings_num
bert_embeddings_num = state_dict["embeddings.word_embeddings.weight"].numpy().size \
                      + state_dict["embeddings.position_embeddings.weight"].numpy().size \
                      + state_dict["embeddings.token_type_embeddings.weight"].numpy().size \
                      + state_dict["embeddings.LayerNorm.weight"].numpy().size \
                      + state_dict["embeddings.LayerNorm.bias"].numpy().size

print(diy_embeddings_num,"--1. diy_embeddings_num") # 16622592
print(bert_embeddings_num,"--1. bert_embeddings_num")



# attention层
num_layers = 12 # transformer层数
num_attention_heads = 12 # 多头个数
q_num = (hidden_size/num_attention_heads) * hidden_size * num_attention_heads + hidden_size # q/k/v参数个数
# k_num/v_num=q_num
attention_output_num = hidden_size * hidden_size + hidden_size # 用一个线性层整合qkv结果
diy_attention_num = q_num *3 +attention_output_num
bert_attention_num = state_dict["encoder.layer.%d.attention.self.query.weight" % 1].numpy().size \
                      + state_dict["encoder.layer.%d.attention.self.query.bias" % 1].numpy().size \
                      + state_dict["encoder.layer.%d.attention.self.key.weight" % 1].numpy().size \
                      + state_dict["encoder.layer.%d.attention.self.key.bias" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.attention.self.value.weight" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.attention.self.value.bias" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.attention.output.dense.weight" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.attention.output.dense.bias" % 1].numpy().size

print(diy_attention_num,"--2. 单层diy_attention_num") # 2362368
print(bert_attention_num,"--2. 单层bert_attention_num")


# attention_out_put_add_norm层
diy_att_norm_num = hidden_size * 2 # 归一化层 wx+b
bert_att_norm_num = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % 1].numpy().size \
                  + state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % 1].numpy().size
print(diy_att_norm_num,"--3. 单层diy_att_norm_num") # 1536
print(bert_att_norm_num,"--3. 单层bert_att_norm_num")

# feed_forward层
diy_feed_forward_num = hidden_size * (hidden_size*4) + (hidden_size*4)  \
                     + (hidden_size*4) * hidden_size + hidden_size # 会先通过一个线性层放大到hidden_size四倍维度；然后再经过一个线性层缩小到hidden_size
bert_feed_forward_num = state_dict["encoder.layer.%d.intermediate.dense.weight" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.intermediate.dense.bias" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.output.dense.weight" % 1].numpy().size \
                     + state_dict["encoder.layer.%d.output.dense.bias" % 1].numpy().size

print(diy_feed_forward_num,"--4. 单层diy_feed_forward_num") # 4722432
print(bert_feed_forward_num,"--4. 单层bert_feed_forward_num")


# out_put_add_norm层
diy_out_put_add_norm = diy_att_norm_num # 和attention的add_norm层参数个数一样
bert_output_norm_num = state_dict["encoder.layer.%d.output.LayerNorm.weight" % 1].numpy().size \
                  + state_dict["encoder.layer.%d.output.LayerNorm.bias" % 1].numpy().size
print(diy_out_put_add_norm,"--5. 单层diy_out_put_add_norm") # 1536
print(bert_output_norm_num,"--5. 单层bert_output_norm_num")

# pooler层
diy_pooler_num = hidden_size * hidden_size + hidden_size
bert_pooler_num = state_dict["pooler.dense.weight"].numpy().size + state_dict["pooler.dense.bias"].numpy().size
print(diy_pooler_num,"--6. diy_pooler_num") # 590592
print(diy_pooler_num,"--6. diy_pooler_num")

# 其中一层transformer的参数量是2+3+4+5

# bert总参数量
diy_bert_num = diy_embeddings_num \
    + (diy_attention_num + diy_att_norm_num + diy_feed_forward_num +diy_out_put_add_norm) * 12 \
    + diy_pooler_num
bert_num = sum(p.numel() for p in state_dict.values())
print(diy_bert_num,"--final. diy_bert_num") # 102267648
print(bert_num,"--final. bert_num") # 102267648
