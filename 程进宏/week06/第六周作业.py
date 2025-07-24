from transformers import BertModel

bert = BertModel.from_pretrained(r"E:\ai_workspace\nlp20\week6\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

print("===== Embedding 各部分参数：")
embedding_total = 0  # 用于累计参数总数

for name, param in state_dict.items():
    if "embeddings.word_embeddings" in name:
        label = "Token Embedding"
    elif "embeddings.position_embeddings" in name:
        label = "Position Embedding"
    elif "embeddings.token_type_embeddings" in name:
        label = "Segment Embedding"
    elif "embeddings.LayerNorm" in name:
        label = "LayerNorm"
    else:
        label = "Other" 

    count = param.numel()
    embedding_total += count
    print(f"{label:<20}: {name:<50} shape={tuple(param.shape)}, params={count:,}")

print("-" * 100)
print(f"{'Embedding Total':<20}: {embedding_total:,} parameters")
