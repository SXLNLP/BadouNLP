# -*- coding: utf-8 -*-
import torch
import logging
from transformers import BertTokenizer

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        
    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.model.eval()
        
        # 这里可以添加具体的评估逻辑
        # 例如生成一些样本文本或计算验证集上的损失
        
        # 示例：生成一些文本
        input_text = "阿根廷歹徒抢服装尺码不对拿回店里换"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.config["device"])
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=self.config["output_max_length"],
                num_beams=self.config["beam_size"],
                early_stopping=True
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info(f"输入: {input_text}")
        self.logger.info(f"生成: {generated_text}")
        
        return
