from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\组件\ppt\AI\数据处理与统计分析\bert-base-chinese", return_dict=False)
config = bert.config  # 获取模型配置参数

# 提取关键参数
hidden_size = config.hidden_size
vocab_size = config.vocab_size
position_size = config.max_position_embeddings
type_size = config.type_vocab_size
num_layers = config.num_hidden_layers
intermediate_size = config.intermediate_size
num_heads = config.num_attention_heads

# 1. 计算嵌入层参数
embedding_params = vocab_size * hidden_size + position_size * hidden_size + type_size * hidden_size

# 2. 单层编码器参数
attention_params = (3 * hidden_size * hidden_size) + hidden_size * hidden_size + 4 * hidden_size
ffn_params = (hidden_size * intermediate_size) + (intermediate_size * hidden_size) + intermediate_size + hidden_size
layer_norm_params = 2 * hidden_size * 2  # 两个LayerNorm层
per_layer_params = attention_params + ffn_params + layer_norm_params

# 3. 池化层参数
pooler_params = hidden_size * hidden_size + hidden_size

# 总参数
total_params = embedding_params + (num_layers * per_layer_params) + pooler_params

print(f"Formula计算结果：{total_params:,}")
print(f"遍历参数验证结果：{sum(p.numel() for p in bert.parameters())}")
