# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional

class SFTModel(nn.Module):
    """SFT模型类，用于监督微调"""
    
    def __init__(self, config: Dict[str, Any]):
        super(SFTModel, self).__init__()
        self.config = config
        
        # 加载预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config["bert_model_path"],
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.get("fp16", False) else torch.float32
        )
        
        # 获取tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_model_path"])
        
        # 添加特殊token
        special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[CLS]',
            'eos_token': '[SEP]'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # 调整模型词汇表大小
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_ids, max_length=128, **kwargs):
        """生成文本"""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                **kwargs
            )
        return outputs
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Dict[str, Any]):
        """从预训练模型加载"""
        model = cls(config)
        model.model = AutoModelForCausalLM.from_pretrained(model_path)
        model.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model

class NewsTitleGenerator:
    """新闻标题生成器"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.model = SFTModel.from_pretrained(model_path, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def generate_title(self, content: str, max_length: int = 128) -> str:
        """根据新闻内容生成标题"""
        # 构建输入文本
        input_text = self.config["prompt_template"].format(content=content)
        
        # 编码输入
        inputs = self.model.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.config["max_length"]
        )
        
        # 移动到设备
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 生成标题
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=self.model.tokenizer.pad_token_id,
                eos_token_id=self.model.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # 解码输出
        generated_ids = outputs[0]
        generated_text = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 提取生成的标题部分
        title = generated_text.replace(input_text, "").strip()
        
        return title
    
    def batch_generate(self, contents: list, max_length: int = 128) -> list:
        """批量生成标题"""
        titles = []
        for content in contents:
            title = self.generate_title(content, max_length)
            titles.append(title)
        return titles 