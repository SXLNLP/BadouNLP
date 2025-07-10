from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\AI\八斗精品班\第六周 语言模型和预训练\bert-base-chinese", return_dict=False)

parameters_total = sum(p.numel() for p in bert.parameters())

state_dict_total = sum(t.numel() for t in bert.state_dict().values())

print(parameters_total, state_dict_total)

# 假设使用float32精度，算一下这些参数的大小验证一下
print(f"模型占用内存计算为{parameters_total * 4 / 1024 ** 2}MB")
