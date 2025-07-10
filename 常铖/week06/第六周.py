from transformers import BertModel

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-chinese')

# 计算可学习参数的数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"BERT模型的可学习参数总数: {total_params}")


