import torch.nn as nn
import os
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModel
from torch.optim import Adam, SGD

def TorchModel():
    """加载BERT模型用于NER任务，强制使用本地文件"""
    model_path = Config["pretrain_model_path"]
    num_labels = Config.get("class_num", 9)  # BIO格式标签数量
    
    print(f"✅ 使用本地BERT模型: {model_path}")
    print(f"🏷️  NER标签数量: {num_labels}")
    
    # 强制使用本地文件，不从网络下载
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True,  # 强制只使用本地文件
        trust_remote_code=True,
        return_dict=True       # 确保返回字典格式
    )
    
    print(f"🎉 NER模型加载成功！参数量: {model.num_parameters():,}")
    return model


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
