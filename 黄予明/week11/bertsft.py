#基于bert的sft的训练
import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader


bertTokenizer = BertTokenizer.from_pretrained("/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese")
bertModel = BertModel.from_pretrained("/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese")

class Bertsftmodel:
    def __init__(self, bertModel, bertTokenizer):
        self.bertModel = bertModel
        self.bertTokenizer = bertTokenizer
        self.liner=nn.Linear(bertModel.config.hidden_size, bertModel.config.vocab_size)
        self.loss=nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, x, attention_mask,y_pred=None):
        if y_pred is None:
            hidden_state, _ = self.bertModel(x, attention_mask=mask) #bertModel的输出是tuple，第一个是hidden_state，第二个是pooler_output
            y_pred = self.liner(hidden_state)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测不用mask，直接用y_pred

            hidden_state= self.bertModel(x)
            return self.liner(hidden_state)

    def generate(self, input_ids, attention_mask):
        return self.bertModel.generate(input_ids, attention_mask)

#处理训练数据 sample_data.json
def load_corpus():
    corpus = []
    with open("sample_data.json", encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append(['cls',line["title"],'sep', line["content"]])
            print(corpus)
            
    return corpus

def create_mask(title, content):

    # 对title和content进行分词
    title_tokens = bertTokenizer.tokenize(title)
    content_tokens = bertTokenizer.tokenize(content)
    
    # 计算长度（包含特殊token）
    len_title = len(title_tokens) + 2  # +2 for [CLS] and [SEP]
    len_content = len(content_tokens) + 1  # +1 for [SEP]
    total_length = len_title + len_content
    
    # 创建掩码张量 (全1矩阵)
    mask = torch.ones(total_length, total_length)
    
    # 设置title部分的掩码规则
    for i in range(len_title):
        # title的token不能看到content的任何token
        mask[i, len_title:] = 0
    
    # 设置content部分的掩码规则  
    for i in range(len_content):
        # content的token不能看到后面的content token (因果掩码)
        mask[len_title + i, len_title + i + 1:] = 0
    
    return mask
    
def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

#构建bertencode，mask和padding  后的张量    
def build_mask_and_padding():
    """
    为整个数据集构建掩码和填充
    返回: 掩码列表和最大长度
    """
    processed_corpus = load_corpus("sample_data.json")
    masks = []
    
    # 为每个样本创建掩码
    for text in processed_corpus:
        # text格式: ['cls', title, 'sep', content]
        title = text[1]  # 标题
        content = text[3]  # 内容
        
        # 创建该样本的掩码
        sample_mask = create_mask(title, content)
        masks.append(sample_mask)
    
    # 计算最大长度
    max_length = max(mask.shape[0] for mask in masks)
    
    # 对所有掩码进行填充到相同长度
    padded_masks = []
    for mask in masks:
        padded_mask = pad_mask(mask, (max_length, max_length))
        padded_masks.append(padded_mask)

    return padded_masks




#构建数据加载器
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
   


#训练模型
def train_model(modBertsftmodelel, train_data_loader, epochs):
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
                if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return
if __name__ == "__main__":
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    max_length = 50       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    corpus = load_corpus()     #加载语料
    train_data = build_dataset(bertTokenizer, corpus, max_length, batch_size)  #建立数据集

    train_model(Bertsftmodel(bertModel, bertTokenizer), build_data_loader("sample_data.json"))
