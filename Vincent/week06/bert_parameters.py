from transformers import BertModel
from collections import defaultdict


bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

print(state_dict)

layer_param_count = defaultdict(int)

for name, param in state_dict.items():
    # 按名称判断属于哪个模块
    if name.startswith("embeddings."):
        layer_param_count["embeddings"] += param.numel()
    elif name.startswith("encoder.layer."):
        # 提取层号，如 encoder.layer.0.attention.self.query.weight
        layer_id = name.split(".")[2]
        layer_param_count[f"encoder.layer.{layer_id}"] += param.numel()
    elif name.startswith("pooler."):
        layer_param_count["pooler"] += param.numel()
    else:
        layer_param_count["other"] += param.numel()  # catch-all

total = 0
for layer_name in sorted(layer_param_count.keys()):
    count = layer_param_count[layer_name]
    total += count
    print(f"{layer_name:20s}: {count:,} parameters")

print(f"\nTotal Parameters: {total:,}")
