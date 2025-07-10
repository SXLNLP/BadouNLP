"""
計算bert-base-chinese模型的參數量
"""

import torch
import json
from transformers import BertModel, BertConfig

def load_config(config_path):
    """載入配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def calculate_bert_params_manually(config):
    """
    根據config.json手動計算BERT模型參數量
    """
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_hidden_layers = config['num_hidden_layers']
    num_attention_heads = config['num_attention_heads']
    intermediate_size = config['intermediate_size']
    max_position_embeddings = config['max_position_embeddings']
    type_vocab_size = config['type_vocab_size']
    
    total_params = 0
    
    # 1. Word Embeddings
    word_embeddings = vocab_size * hidden_size
    total_params += word_embeddings
    
    # 2. Position Embeddings
    position_embeddings = max_position_embeddings * hidden_size
    total_params += position_embeddings
    
    # 3. Token Type Embeddings
    token_type_embeddings = type_vocab_size * hidden_size
    total_params += token_type_embeddings
    
    # 4. Embedding LayerNorm
    embedding_layer_norm = hidden_size * 2  # weight + bias
    total_params += embedding_layer_norm
    
    # 5. 每個Transformer層的參數
    for layer in range(num_hidden_layers):
        # 5.1 自注意力層 (Self-Attention)
        # Query, Key, Value 矩陣
        qkv_params = 3 * hidden_size * hidden_size
        # 輸出投影矩陣
        output_projection = hidden_size * hidden_size
        # 注意力層歸一化
        attention_layer_norm = hidden_size * 2
        
        # 5.2 前饋網路 (Feed-Forward Network)
        # 第一個線性層
        ff1 = hidden_size * intermediate_size
        # 第二個線性層
        ff2 = intermediate_size * hidden_size
        # 前饋網路層歸一化
        ff_layer_norm = hidden_size * 2
        
        layer_params = qkv_params + output_projection + attention_layer_norm + ff1 + ff2 + ff_layer_norm
        total_params += layer_params
    
    # 6. 池化層 (Pooler)
    pooler_params = hidden_size * hidden_size + hidden_size  # weight + bias
    total_params += pooler_params
    
    return total_params

def calculate_bert_params_automatically(model):
    """
    使用PyTorch自動計算模型參數量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """格式化數字顯示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def main():
    try:
        """
        假設本地上有bert-base-chinese模型
        """
        model = BertModel.from_pretrained('./bert-base-chinese')
        print("✓ 成功載入本地BERT模型")
    except Exception as e:
        raise Exception(f"無法載入BERT模型: {e}")
    
    # 方法2: 載入配置檔案
    config = load_config('./bert-base-chinese/config.json')
    print(f"\n配置檔案資訊:")
    print(f"  - 詞彙表大小: {config['vocab_size']:,}")
    print(f"  - 隱藏層大小: {config['hidden_size']}")
    print(f"  - 隱藏層數量: {config['num_hidden_layers']}")
    print(f"  - 注意力頭數: {config['num_attention_heads']}")
    print(f"  - 中間層大小: {config['intermediate_size']}")
    print(f"  - 最大位置編碼: {config['max_position_embeddings']}")
    
    # 手動計算參數量
    manual_params = calculate_bert_params_manually(config)
    print(f"\n手動計算參數量: {manual_params:,} ({format_number(manual_params)})")
    
    # 自動計算參數量
    total_params, trainable_params = calculate_bert_params_automatically(model)
    print(f"自動計算參數量: {total_params:,} ({format_number(total_params)})")
    print(f"可訓練參數量: {trainable_params:,} ({format_number(trainable_params)})")
    
if __name__ == "__main__":
    main()

