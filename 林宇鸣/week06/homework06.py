import torch
from transformers import BertModel

# 加载预训练的 BERT 模型
bert = BertModel.from_pretrained("bert-base-chinese")
state_dict = bert.state_dict()

# 计算总的参数量通过 state_dict
total_params_state_dict = sum(param.numel() for param in state_dict.values())
print(f"Total number of parameters from state_dict: {total_params_state_dict}")

# def calculate_bert_param_memory(state_dict):
#     total_memory = 0
#     for key, tensor in state_dict.items():
#         num_elements = tensor.nlement()
#         element_size = tensor.element_size()
#         memory_usage = num_elements * element_size
#         total_memory += memory_usage
#         print(f"Memory usage for {key}: {memory_usage} B")
#         # print(f"\nTotal memory usage: {total_memory} B")
#         return total_memory
# 定义公式计算的函数
def calculate_bert_params(vocab_size, hidden_size, num_layers, intermediate_size, max_position_embeddings,
                          type_vocab_size):
    # 嵌入层参数
    embedding_params = (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size

    # Transformer 层参数计算
    attention_params_per_layer = (4 * hidden_size ** 2 + 3 * hidden_size + 2 * hidden_size)
    feed_forward_params_per_layer = (hidden_size * intermediate_size + intermediate_size +
                                     intermediate_size * hidden_size + hidden_size +
                                     2 * hidden_size)

    transformer_layer_params = attention_params_per_layer + feed_forward_params_per_layer

    # 总参数量
    total_params = embedding_params + num_layers * transformer_layer_params

    return total_params




# BERT 的一些超参数（根据中文 BERT 模型的设置）
vocab_size = 21128  # 词汇表大小
hidden_size = 768  # 隐藏层大小
num_layers = 12  # Transformer 层数
intermediate_size = 3072  # 前馈网络的中间层大小
max_position_embeddings = 512  # 最大位置嵌入数
type_vocab_size = 2  # 类型词汇表大小

# 计算总参数量通过公式
total_params_formula = calculate_bert_params(vocab_size, hidden_size, num_layers,
                                             intermediate_size, max_position_embeddings,
                                             type_vocab_size)

print(f"Total number of parameters calculated using formula: {total_params_formula}")

# 对比
if total_params_state_dict == total_params_formula:
    print("Both methods yield the same total parameter count.")
else:
    print("There is a discrepancy between the two methods.")


# # 计算总参数内存
# memory_usage = calculate_bert_param_memory(state_dict)
# print(f"Total memory usage: {memory_usage} B")
# Total memory usage: 409070592 B
# def human_readable_size(size, decimal_places=2):
#     for unit in ['B', 'KB', 'MB', 'GB']:
#         if size < 1024:
#             return f"{size:.{decimal_places}f} {unit}"
#         size /= 1024
#     return f"{size:.{decimal_places}f} TB"
#
# print(f"Total memory usage: {human_readable_size(memory_usage)}")
