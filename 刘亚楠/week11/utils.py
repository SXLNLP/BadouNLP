import os
import torch
from transformers import BertTokenizer
from config import config
from model import Title2ContentModel
from transformers import BertLMHeadModel, BertTokenizer,BertConfig

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({'additional_special_tokens': [config.bos_token, config.eos_token]})
    return tokenizer

def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model(model_path):
    # 加载配置时指定自定义类
    #config = BertConfig.from_pretrained(model_path)
    #model = Title2ContentModel(config)  # 用正确的类初始化

    # 加载权重
    #state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    #model.load_state_dict(state_dict)

    # 加载tokenizer
    #tokenizer = BertTokenizer.from_pretrained(model_path)

    model = Title2ContentModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer
