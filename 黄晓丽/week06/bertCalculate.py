import json
from pathlib import Path

def calculate_bert_chinese_params():
    config_path = Path(r"D:\BaiduNetdiskDownload\aaabadouDownload\bert-base-chinese\config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 从配置中提取参数
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    intermediate_size = config["intermediate_size"]
    max_position_embeddings = config["max_position_embeddings"]
    type_vocab_size = config["type_vocab_size"]

    #计算Embedding层参数量
    token_embedding_params = vocab_size * hidden_size
    position_embedding_params = max_position_embeddings * hidden_size
    segment_embedding_params = type_vocab_size * hidden_size
    embedding_total = token_embedding_params + position_embedding_params + segment_embedding_params

    # 2. 计算每个Transformer层的参数量
    # 注意力机制参数
    attention_params = 0
    # Q/K/V投影矩阵 (每个头)
    qkv_params_per_head = 3 * (hidden_size // num_attention_heads) * hidden_size
    # 所有头
    qkv_params_total = num_attention_heads * qkv_params_per_head
    # 输出投影
    output_proj_params = hidden_size * hidden_size
    # 偏置 (通常有)
    bias_params = 3 * hidden_size + hidden_size  # Q/K/V的偏置 + 输出偏置

    attention_params = qkv_params_total + output_proj_params + bias_params

    # 前馈神经网络参数
    ffn_params = (hidden_size * intermediate_size) + intermediate_size  # 第一层 (权重 + 偏置)
    ffn_params += (intermediate_size * hidden_size) + hidden_size  # 第二层 (权重 + 偏置)

    # 层归一化参数 (两个归一化层)
    norm_params = 2 * 2 * hidden_size  # 每个归一化层有gamma和beta参数

    # 每层总参数
    per_layer_params = attention_params + ffn_params + norm_params

    # 3. 所有Transformer层总参数
    all_layers_params = num_hidden_layers * per_layer_params

    # 4. Pooler层参数 ([CLS]输出)
    pooler_params = hidden_size * hidden_size + hidden_size  # 权重 + 偏置

    # 5. 总参数量
    total_params = embedding_total + all_layers_params + pooler_params

    print("\n" + "=" * 60)
    print(f"BERT 参数量计算 (模型: {config.get('_name_or_path', 'bert-base-chinese')})")
    print("=" * 60)

    print("\n[配置参数]")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_hidden_layers: {num_hidden_layers}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_position_embeddings: {max_position_embeddings}")
    print(f"  type_vocab_size: {type_vocab_size}")

    print("\n[Embedding 层]")
    print(f"  Token Embeddings: {format_number(token_embedding_params)}")
    print(f"  Position Embeddings: {format_number(position_embedding_params)}")
    print(f"  Segment Embeddings: {format_number(segment_embedding_params)}")
    print(f"  → Embedding 总计: {format_number(embedding_total)}")

    print("\n[Transformer 层]")
    print(f"  单层参数: {format_number(per_layer_params)}")
    print(f"  {num_hidden_layers} 层总计: {format_number(all_layers_params)}")

    print("\n[Pooler 层]")
    print(f"  Pooler 层: {format_number(pooler_params)}")

    print("\n" + "=" * 60)
    print(f"总参数量: {format_number(total_params)}")
    print("=" * 60)

    # 计算各部分占比
    total = total_params
    embed_pct = embedding_total / total * 100
    layers_pct = all_layers_params / total * 100
    pooler_pct = pooler_params / total * 100

    print("\n[各部分占比]")
    print(f"  Embedding 层: {embed_pct:.2f}%")
    print(f"  Transformer 层: {layers_pct:.2f}%")
    print(f"  Pooler 层: {pooler_pct:.2f}%")

    # 估算模型大小
    # 假设使用float32 (4字节/参数)
    model_size_mb = (total * 4) / (1024 ** 2)
    print(f"\n估算模型大小 (float32): {model_size_mb:.2f} MB")

def format_number(num):
    """格式化大数字显示"""
    return f"{num:,}"

if __name__ == "__main__":
    calculate_bert_chinese_params()
