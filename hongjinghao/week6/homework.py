from transformers import BertModel

'''
作业要求：
计算bert中所有可训练的数字有多少，即所有可训练的矩阵进行 行乘列 计算，获得所有的可训练向量维度，并且将其相加

bert具体实现过程可见笔记内容

bert-base-chinese的路径需要按照自己实际路径来
东西可以自己去谷歌下载(https://huggingface.co/google-bert/bert-base-chinese)，或者网盘里面也有
'''

bert = BertModel.from_pretrained(r"D:\study\ai\录播\week6\bert-base-chinese\bert-base-chinese", return_dict=False)
conf = bert.config.to_dict()
# embedding

token_embeddings_size = conf["vocab_size"] * conf["hidden_size"]
segment_embeddings_size = conf["type_vocab_size"] * conf["hidden_size"]
position_embeddings_size = conf["max_position_embeddings"] * conf["hidden_size"]
w_layer_norm_size = conf["hidden_size"]
b_layer_norm_size = conf["hidden_size"]
embedding_sum = token_embeddings_size + segment_embeddings_size + position_embeddings_size + w_layer_norm_size + b_layer_norm_size

# transformer - 1 self-attention
# 多头机制不影响矩阵的大小。仅是按照头的数量，对矩阵按列做对应的切分，分别训练后再拼接
# 多头拼接以后还会经过一个线性层，因此还有一个输出时的线性层，名为output.dense。在transformer中线性层被称为dense，在pytorch中线性层被称为linear
wq_size = conf["hidden_size"] * conf["hidden_size"]
bq_size = conf["hidden_size"]

wk_size = conf["hidden_size"] * conf["hidden_size"]
bk_size = conf["hidden_size"]

wv_size = conf["hidden_size"] * conf["hidden_size"]
bv_size = conf["hidden_size"]

wop_linear_size = conf["hidden_size"] * conf["hidden_size"]
bop_linear_size = conf["hidden_size"]

self_attention_sum = wq_size + bq_size + wk_size + bk_size + wv_size + bv_size + wop_linear_size + bop_linear_size

# transformer - 2 Add&Normalize
# 残差本质就是将原有输入与上一模块的输出矩阵进行矩阵加，将加合结果经过LayerNorm层，因此LayerNorm层也是可训练的一部分
w_after_self_attention_layerNorm = conf["hidden_size"]
b_after_self_attention_layerNorm = conf["hidden_size"]

after_self_attention_layerNorm_sum = w_after_self_attention_layerNorm + b_after_self_attention_layerNorm

# transformer - 3 Feed Forward
# 公式 Liner(gelu(Liner(x))) 涉及两层线性层，激活层是不需要训练。并且需要注意的是，这中间有一个升维和降维的过程，因此线性层并非hidden_size
wl1_size = conf["hidden_size"] * conf["intermediate_size"]
bl1_size = conf["intermediate_size"]

wl2_size = conf["intermediate_size"] * conf["hidden_size"]
bl2_size = conf["hidden_size"]

ff_sum = wl1_size + bl1_size + wl2_size + bl2_size

# transformer - 4 Add&Normalize
# 残差本质就是将原有输入与上一模块的输出矩阵进行矩阵加，将加合结果经过LayerNorm层，因此LayerNorm层也是可训练的一部分
w_after_ff_layerNorm = conf["hidden_size"]
b_after_ff_layerNorm = conf["hidden_size"]

after_ff_layerNorm_sum = w_after_ff_layerNorm + b_after_ff_layerNorm

# transformer 会过 n层，需要将结果*n
transformers_sum = conf["num_hidden_layers"] * (ff_sum + self_attention_sum + after_ff_layerNorm_sum +
                                                after_self_attention_layerNorm_sum)

# pooler 上课没讲过，但是实际bert中还有个pooler层
# 作用是训练[CLS]向量，即句子开头前的一个向量，这个向量用于表示一整句话的意思。pooler层就是过一个线性层，然后在过一个tan激活层
w_pooler_size = conf["hidden_size"] * conf["hidden_size"]
b_pooler_size = conf["hidden_size"]
pooler_sum = w_pooler_size + b_pooler_size

# 自行分析计算
output = transformers_sum + embedding_sum +pooler_sum
print(output)

# 直接遍历bert中可训练的矩阵/向量，用于校验自行分析是否正确
total_params = 0
for name, param in bert.named_parameters():
    if param.requires_grad:  # 只统计可训练参数
        shape = param.shape
        count = param.numel()  # 元素数量 = 行×列×...
        total_params += count
print(f"\nTotal trainable parameters: {total_params}")
