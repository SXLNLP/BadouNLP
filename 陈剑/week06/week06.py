from transformers import BertModel

bert = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()


total_elements = 0
for key, tensor in state_dict.items():
    total_elements += tensor.numel()

print("所有矩阵元素总数:", total_elements)
