
from transformers import BertModel


model = BertModel.from_pretrained('bert-base-chinese')
for name, param in model.named_parameters():
    print(name, param.shape)

total_params = sum(p.numel() for p in model.parameters())
print("Bert参数总量: ", total_params)


max_length = 512
hide_size = 768
intermediate_size = 3072
vocab_size = 21128
senten_count = 2

#embedding
embedding_count = vocab_size * hide_size + senten_count * hide_size + max_length * hide_size + hide_size + hide_size

#self-attention
attention_count = 3 * (hide_size * hide_size + hide_size) + hide_size * hide_size + hide_size

#layer-norm-1
layer_norm_count1 = hide_size + hide_size

#FeedForward
feed_forward_count = hide_size * intermediate_size + intermediate_size + intermediate_size * hide_size + hide_size

#layer-norm-2
layer_norm_count2 = hide_size + hide_size

#pooler
pooler_count = hide_size * hide_size + hide_size

total_count = embedding_count + attention_count + layer_norm_count1 + feed_forward_count + layer_norm_count2 + pooler_count
print("个人计算所得参数总量: ",total_count)
