#coding:utf8

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
week2的例子，修改引入bert
"""
import torch
from transformers import BertModel

# 计算总内存占用
def calculate_bert_param_memory(state_dict):
    total_memory = 0
    for key, tensor in state_dict.items():
        num_elements = tensor.numel()
        element_size = tensor.element_size()  # 单个元素占用的字节数
        memory_usage = num_elements * element_size
        total_memory += memory_usage
        print(f"Memory usage for {key}: {memory_usage} B")
    return total_memory

# 内存格式化
def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} TB"

# 公式计算参数量
def calculate_bert_params(vocab_size, hidden_size, num_layers, intermediate_size, max_position_embeddings, type_vocab_size):
    # Embedding 参数
    embedding_params = (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size

    # Transformer 每层参数
    attention_params_per_layer = (4 * hidden_size ** 2 + 3 * hidden_size + 2 * hidden_size)
    feed_forward_params_per_layer = (hidden_size * intermediate_size + intermediate_size + intermediate_size * hidden_size + hidden_size + 2 * hidden_size)

    transformer_layer_params = attention_params_per_layer + feed_forward_params_per_layer

    total_params = embedding_params + num_layers * transformer_layer_params
    return total_params

# 主程序
def main():
    # 加载预训练的 BERT 中文模型
    bert = BertModel.from_pretrained("bert-base-chinese")
    state_dict = bert.state_dict()

    # 计算总参数量（通过 state_dict 真实统计）
    total_params_state_dict = sum(param.numel() for param in state_dict.values())
    print(f"Total number of parameters from state_dict: {total_params_state_dict}")

    # BERT 中文模型参数配置
    vocab_size = 21128
    hidden_size = 768
    num_layers = 12
    intermediate_size = 3072
    max_position_embeddings = 512
    type_vocab_size = 2

    # 公式计算参数量
    total_params_formula = calculate_bert_params(vocab_size, hidden_size, num_layers, intermediate_size, max_position_embeddings, type_vocab_size)
    print(f"Total number of parameters calculated using formula: {total_params_formula}")

    # 比较两种计算方式
    if total_params_state_dict == total_params_formula:
        print("Both methods yield the same total parameter count.")
    else:
        print("There is a discrepancy between the two methods.")

    # 计算总内存占用
    memory_usage = calculate_bert_param_memory(state_dict)
    print(f"Total memory usage: {human_readable_size(memory_usage)}")

if __name__ == "__main__":
    main()
