from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''
bert = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese\bert-base-chinese", return_dict=False)

def calculate_bert_param_memory(state_dict):
    total_memory = 0
    for key, tensor in state_dict.items():
        num_elements = tensor.nelement()
        element_size = tensor.element_size()
        memory_usage = num_elements * element_size
        total_memory += memory_usage
        print(f"{key}: {memory_usage} B")
    # print(f"\nTotal memory usage: {total_memory} B")
    return total_memory

# 获取 BERT 模型的 state_dict
state_dict = bert.state_dict()

# 计算总参数内存
memory_usage = calculate_bert_param_memory(state_dict)
print(f"Total memory usage: {memory_usage} B")
# Total memory usage: 409070592 B
def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} TB"
  
print(f"Total memory usage: {human_readable_size(memory_usage)}")
# Total memory usage: 390.12 MB
