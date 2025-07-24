import torch
from transformers import BertModel

# 加载模型
bert = BertModel.from_pretrained(
    r"D:\BaiduYunDownload\八斗精品班\第六周 语言模型 （预习-rVX5mB9h\bert-base-chinese")
# state_dict = bert.state_dict()

# 获取所有可训练参数
untrained_params = [(name, param) for name, param in bert.named_parameters()
                    if param.requires_grad]

# 打印每个参数的名字和形状
print("可训练的参数名称和形状如下：")
for name, param in untrained_params:
    # param.size()获取的形状param.shape是一个torch.size对象，是torch.Size([768, 768])
    # print(f"{name:<10} {param.size()}")  # 左对齐10字符，对形状强制以tuple输出
    print(f"{name:<60} {tuple(param.size())}")  # 左对齐10字符，对形状强制以tuple输出

# 计算所有可训练参数的总量
total_params = sum(p.numel() for _, p in untrained_params)
print(f"所有可训练的参数量有{total_params}个")
print(f"整体大小约为{total_params / 1e6:.2f}M个")
