from transformers import BertModel
import torch

# 加载BERT模型（默认使用bert-base-uncased）
model = BertModel.from_pretrained("bert-base-uncased")

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"BERT总参数: {total_params:,}")

# 分层统计参数量
embed_params = sum(p.numel() for p in model.embeddings.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
pooler_params = sum(p.numel() for p in model.pooler.parameters()) if model.pooler else 0

print("\n分层统计:")
print(f"- Embedding层: {embed_params:,} ({embed_params/total_params:.1%})")
print(f"- Transformer层: {encoder_params:,} ({encoder_params/total_params:.1%})")
print(f"- Pooler层: {pooler_params:,} ({pooler_params/total_params:.1%})")
print(f"= 合计: {embed_params + encoder_params + pooler_params:,}")
