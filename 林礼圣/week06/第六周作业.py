import torch
import os
import zipfile
from transformers import BertModel

# 解压模型文件（如果尚未解压）
model_zip = "bert-base-chinese.zip"
model_dir = "bert-base-chinese"

if not os.path.exists(model_dir):
    with zipfile.ZipFile(model_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    print(f"已解压模型到: {model_dir}")

# 加载模型
model = BertModel.from_pretrained(model_dir)
print("模型加载成功！")

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\nBERT-base-chinese 总参数量: {total_params:,}")

# 详细参数分解计算
print("\n详细参数分解:")
embedding_params = 0
encoder_params = 0
pooler_params = 0

# 1. 嵌入层
embeddings = model.embeddings
embedding_params += sum(p.numel() for p in embeddings.word_embeddings.parameters())  # 词嵌入
embedding_params += sum(p.numel() for p in embeddings.position_embeddings.parameters())  # 位置嵌入
embedding_params += sum(p.numel() for p in embeddings.token_type_embeddings.parameters())  # 类型嵌入
embedding_params += sum(p.numel() for p in embeddings.LayerNorm.parameters())  # 层归一化

# 2. 编码器层
encoder = model.encoder
for layer in encoder.layer:
    # 自注意力层
    attn = layer.attention.self
    encoder_params += sum(p.numel() for p in attn.query.parameters())
    encoder_params += sum(p.numel() for p in attn.key.parameters())
    encoder_params += sum(p.numel() for p in attn.value.parameters())
    
    # 注意力输出层
    attn_output = layer.attention.output
    encoder_params += sum(p.numel() for p in attn_output.dense.parameters())
    encoder_params += sum(p.numel() for p in attn_output.LayerNorm.parameters())
    
    # 前馈网络
    ff = layer.intermediate
    encoder_params += sum(p.numel() for p in ff.dense.parameters())
    
    ff_output = layer.output
    encoder_params += sum(p.numel() for p in ff_output.dense.parameters())
    encoder_params += sum(p.numel() for p in ff_output.LayerNorm.parameters())

# 3. 池化层
pooler = model.pooler
pooler_params += sum(p.numel() for p in pooler.dense.parameters())

# 验证计算结果
calculated_total = embedding_params + encoder_params + pooler_params
print(f"嵌入层参数: {embedding_params:,}")
print(f"编码器参数: {encoder_params:,} (12层)")
print(f"池化层参数: {pooler_params:,}")
print(f"计算总参数量: {calculated_total:,}")
print(f"实际总参数量: {total_params:,}")
print(f"验证结果: {'一致' if calculated_total == total_params else '不一致'}")
