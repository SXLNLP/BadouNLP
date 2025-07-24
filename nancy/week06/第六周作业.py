# 计算Bert参数量
from transformers import BertModel

# 加载Bert模型
bert = BertModel.from_pretrained("./bert-base-chinese", return_dict=False)

# 获取所有参数
state_dict = bert.state_dict()

total_params = 0
print("各参数名称及其形状:")
for name, param in state_dict.items():
    print(name,":", tuple(param.shape))
    total_params += param.numel()

print("\nBert模型参数总量: ",total_params) 
