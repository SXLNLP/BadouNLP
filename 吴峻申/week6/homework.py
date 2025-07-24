from transformers import BertModel

model = BertModel.from_pretrained("bert-base-chinese")
total_params = sum(p.numel() for p in model.parameters())
print(total_params)  # 输出 102,267,648
