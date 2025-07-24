"""通过state_dict返回的字典自己编写程序计算bert模型的参数量"""
from transformers import BertModel
bert = BertModel.from_pretrained(r"bert-base-chinese")
state_dict = bert.state_dict()
#print(state_dict.keys()) #查看state_dict返回的字典的键
all_params = 0
for key, value in state_dict.items():
    print(f"键{key}对应形状是{value.shape}")
    total_params = 1
    for d in value.shape:
        total_params *= d
    all_params += total_params
    print(f"键{key}对应参数量是{total_params},数据类型是{value.dtype},数据大小是{value.nbytes/1024/1024}MB") 
print(f"bert模型的参数量是{all_params}")
