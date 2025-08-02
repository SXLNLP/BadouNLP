#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
"""
基于pytorch的LSTM语言模型
"""


class SFTLanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path,sep_token_id=102):
        super(SFTLanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy
        self.sep_token_id = sep_token_id #[sep]token作为分隔符

    def create_sft_mask(self,input_ids):
        batch_size, seq_len = input_ids.shape
        # print(f"[DEBUG] Creating mask for batch_size={batch_size}, seq_len={seq_len}")
        # 边界检查1：验证输入形状
        assert len(input_ids.shape) == 2, f"Input_ids should be 2D tensor, got {input_ids.shape}"
        mask = torch.zeros(batch_size, seq_len, seq_len)
        #找到分隔符位置
        sep_positions = (input_ids == self.sep_token_id). nonzero()
        # print(f"[INFO] Found {len(sep_positions)} separator positions in total")
        # 边界检查2：验证分隔符位置张量
        if sep_positions.numel() > 0 and len(sep_positions.shape) != 2:
            raise ValueError(f"Invalid sep_positions shape: {sep_positions.shape}")
        if len(sep_positions) == 0:
            return torch.ones_like(mask)

        first_sep_pos = {}
        for pos in sep_positions:
            sample_idx = pos[0].item()
            if sample_idx not in first_sep_pos:
                first_sep_pos[sample_idx] = pos[1].item()
        for i in range(batch_size):
            if i in first_sep_pos:
                # print(f"[DEBUG] Processing sample {i} with sep_pos={first_sep_pos[i]}")
                sep_pos = min(first_sep_pos[i], seq_len-2)
                mask[i, :sep_pos+1, :sep_pos+1] = 1
                mask[i, sep_pos+1:, :sep_pos+1] = 1#文章可见标题
                content_len = seq_len - sep_pos - 1
                if content_len > 0:
                    mask[i, sep_pos+1:, sep_pos+1:] = torch.tril(
                    torch.ones(content_len, content_len))
            else:
                # print(f"[DEBUG] Sample {i} has no separator, using full mask")
                mask[i] = torch.ones(seq_len,seq_len) #没有分隔符样本使用全mask
        # print(f"[INFO] Mask generation completed for batch")
        return mask

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, labels=None):
        if labels is not None:
            batch_size = input_ids.size(0)
            mask = self.create_sft_mask(input_ids)
            if torch.cuda.is_available():
                mask = mask.cuda()
            outputs, _ = self.bert(input_ids, attention_mask=mask)
            logits = self.classify(outputs)
            #只计算文章部分的loss
            sep_mask = (input_ids == self.sep_token_id)
            total_loss = 0
            valid_samples = 0
            for i in range(batch_size):
                if sep_mask[i].any():
                    sep_pos = sep_mask[i].nonzero()[0].item()
                    content_len = input_ids.size(1) - sep_pos - 1
                    if content_len >= 1:
                        content_logits = logits[i, sep_pos+1:]
                        content_labels = labels[i, sep_pos+1:]
                        print(
                            f"Sample {i} - sep_pos: {sep_pos} | content_len: {content_len} | logits_shape: {content_logits.shape} | labels_shape: {content_labels.shape}")
                        if content_logits.size(0) == content_labels.size(0):
                            loss = self.loss(
                                content_logits.view(-1, content_logits.size(-1)),
                                content_labels.view(-1)
                            )
                            total_loss += loss
                            valid_samples += 1
                else:
                    loss = self.loss(
                        logits[i].view(-1,logits.size(-1)),labels[i].view(-1))
                    total_loss += loss
                    valid_samples += 1
            if torch.isnan(total_loss):
                print(f"NaN detected in batch {i}")
                return torch.tensor(0.0, device=input_ids.device)
            return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0)
        else:
            return self.geenerate(input_ids)

#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path,'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            corpus.append(line)
            # prepare_data(title, content)
        return corpus


#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sft_sample(tokenizer, title, content, max_length = 128):
    min_content_length = 10
    if len(content) < min_content_length:
        return None,None
    #用[sep]连接标题和文章
    text = title + "[SEP]" + content
    inputs = tokenizer(
        text, padding = 'max_length',
        max_length=max_length,
        truncation=True,
        return_tensors = 'pt'
    )
    labels = inputs['input_ids'].clone()
    #将标题部分的label设为-100
    sep_pos = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero()
    if sep_pos.numel() == 0:# 添加分隔符检查
        return None,None

    else:
        sep_pos = sep_pos[0].item()
    #确保内容长度有效
    for i in range(labels.size(0)):
        labels[i, :sep_pos+1] = -100
    if inputs['input_ids'].size(1) - sep_pos - 1 < 10:
        return None,None
    return inputs['input_ids'], labels


#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        title, content = random.choice(corpus)
        #如果文章太长，截取window_size长度
        if len(content) > window_size:
            start = random.randint(0, len(content) - window_size)
            content = content[start:start + window_size]
        x, y = build_sft_sample(tokenizer,title, content)
        if x is None:  # 跳过无效样本
            continue
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = SFTLanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(model, tokenizer, title,max_length=512,temperature = 1.0):
    input_text = title + "[SEP]"
    input_ids = tokenizer.encode(input_text,return_tensors='pt')

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    sep_pos = (input_ids[0] == tokenizer.sep_token_id).nonzero()
    if len(sep_pos) == 0:
        sep_pos = len(input_ids[0]) - 1
    else:
        sep_pos = sep_pos[0,0]

    generated = []
    for _ in range(max_length):
        mask = model.create_sft_mask(input_ids).to(input_ids.device)
        outputs, _ = model.bert(input_ids, attention_mask=mask)
        logits = model.classify(outputs[:, -1, :]) / temperature

        top_k = 50
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        # filter_value = -float('Inf')
        # probs = torch.where(probs < 0.01,torch.tensor(filter_value),probs)
        next_token = top_indices[0, torch.multinomial(top_probs,num_samples=1)]

        input_ids = torch.cat([input_ids, next_token.view(1,1)], dim=-1)
        if next_token.item() == tokenizer.sep_token_id or len(generated) > max_length:
            break
        generated.append(next_token.item())
    if generated:
        generated_text = tokenizer.decode(generated,skip_special_tokens=True)
        generated_text = re.sub(r'[^\u4e00-\u9fa5，。！？、：；（）【】《》\s]', '', generated_text)
        return generated_text.strip()
    return "未能生成有效内容"


    # 后处理：移除重复文本和无效字符
    # generated_text = re.sub(r'[\s]+', ' ', generated_text).strip()
    # return generated_text
# def sampling_strategy(logits, tokenizer):
#     if random.random() > 0.1:
#         next_token = torch.argmas(logits, dim=-1, keepdim=True)
#     else:
#         probs = torch.softmax(logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1)
#     input_ids = torch.cat([input_ids, next_token], dim=-1)
#     if next_token.item() == tokenizer.sep_token_id:
#         break
#     return input_ids



def train(corpus_path, pretrain_model_path,save_weight=False):
    epoch_num = 5        #训练轮数
    batch_size = 128       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 128       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 1e-5  #学习率

    #用于存储每轮训练结果的列表
    epoch_losses = []
    epoch_avg_losses = []

    pretrain_model_path = r"E:\09-python\04-八斗课件\第六周 语言模型\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    # title_article_pairs = load_corpus(corpus_path)
    print("\nEpoch\tTrain Losss\tAvg Loss")
    print("-----------")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        batch_count = 0
        random.shuffle(corpus)
        for i in range(0, min(len(corpus), train_sample),batch_size):
            batch_data = corpus[i:i+batch_size]
            input_ids_list=[]
            labels_list=[]
            for item in batch_data:
                title = item["title"]
                content = item["content"]
                input_ids, labels = build_sft_sample(
                    tokenizer,
                    title,
                    content
                )
                if input_ids is not None and labels is not None:
                    input_ids_list.append(input_ids)
                    labels_list.append(labels)
            if not input_ids_list:
                continue
            input_ids = torch.stack(input_ids_list).squeeze(1)
            labels = torch.stack(labels_list).squeeze(1)
            print(f"Batch {i} - Valid labels ratio: {(labels != -100).float().mean().item():.2f}")
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(total_loss)
            epoch_avg_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} 平均loss: {avg_loss:.4f}")
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()

            optim.zero_grad()    #梯度归零
            loss = model(input_ids, labels)   #计算loss
            if not torch.isnan(loss) and loss.item() != 0:
                loss.backward()      #计算梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) #梯度裁剪
                optim.step()         #更新权重
                total_loss += loss.item()
                batch_count += 1
            else:
                print(f"Warning: Invalid loss value {loss.item()} at batch {i}")
            #每10个batch打印一次进度
            if batch_count % 50 == 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch+1} - Batch {batch_count} Loss: {avg_loss:.4f}")
                if batch_count % 200 == 0:
                    test_title = random.choice(corpus)["title"]
                    generated = generate_sentence(model,tokenizer,test_title,max_length=window_size)
                    print(f"示例生成 - 标题： {test_title} \n生成内容: {generated}\n")
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(total_loss)))
            #每个epoch结束后评估
    # avg_epocch_loss = total_loss / batch_count
    # print(f"==========\nEpoch {epoch+1} 平均loss: {avg_epocch_loss:.4f}")
    #生成示例
    # test_titles = ["阿根廷歹徒抢服装尺码不对拿回店里换"]
    # for title in test_titles:
    #     generated = generate_sentence(model, tokenizer, title, max_length=window_size)
    #     print(f"标题： {title} \n生成文章： {generated}\n")
    print("\nTraining Summary:")
    print("Epoch\tTotal Loss\tAvg Loss")
    print("-----------")
    for i in range(epoch_num):
        print(f"{i+1}\t{epoch_losses[i]:.4f}\t{epoch_avg_losses[i]:.4f}")
    if save_weight:
        base_name = os.path.basename(corpus_path).replace("json", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
    else:
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
train(r"E:/09-python/04-八斗课件/第十一周 大语言模型相关1/homework/corpus.json", False)

