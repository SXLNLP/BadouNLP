# coding:utf8
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
import math
import random
import os
import re

"""
基于pytorch的BERT自回归语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        y_pred = self.classify(sequence_output)

        if labels is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), labels.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            vocab[line] = i
    return vocab

class MyDataset(Dataset):
    def __init__(self, corpus_path, vocab_path, max_length):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_length = max_length
        self.data = []
        with open(corpus_path, 'r', encoding='gb2312',errors='ignore') as f:
            text = f.read()
        # 使用正则表达式按句子分割
        sentences = re.split(r'[。？！]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 1:
                # BERT的输入格式是 [CLS] sentence [SEP]
                # 将每个句子处理成适合自回归任务的形式
                # input: [CLS] w1 w2 w3 ... wn [SEP]
                # label: w1 w2 w3 ... wn [SEP] <pad>
                # 在计算loss时，将label向右移动一位
                encoded = self.tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].squeeze(0)
                attention_mask = encoded['attention_mask'].squeeze(0)

                # 创建label，将input_ids向左移动一位，并用-1填充末尾
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = -1 # CrossEntropyLoss会忽略-1

                self.data.append((input_ids, attention_mask, labels))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    #参数设置
    epoch_num = 20        #训练轮数
    batch_size = 16       #每次训练的样本个数
    max_length = 50       #样本最大长度
    learning_rate = 1e-5  #学习率
    vocab_path = r"C:\Users\cj783\Desktop\AI算法工程师\week10\第十周\week10 文本生成问题\week10 文本生成问题\lstm语言模型生成文本\vocab.txt"
    corpus_path = r"C:\Users\cj783\Desktop\AI算法工程师\week10\第十周\week10 文本生成问题\week10 文本生成问题\lstm语言模型生成文本\corpus.txt"
    model_save_path = r"C:\Users\cj783\Desktop\AI算法工程师\week6\week6 语言模型和预训练\bert-base-chinese\pytorch_model.bin"

    # 加载词汇表
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    vocab_size = len(tokenizer.vocab)

    # 建立模型
    model = LanguageModel(vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 数据集
    dataset = MyDataset(corpus_path, vocab_path, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = len(dataloader) * epoch_num
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("模型结构：", model)
    print("开始训练...")

    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epoch_num}, Step {i}/{len(dataloader)}, Loss: {loss.item()}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished, Average Loss: {avg_loss}")

        torch.save(model.state_dict(), model_save_path)

    print("训练完成。")

# 文本生成函数
def generate_text(model, tokenizer, start_text, max_len=50):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(input_ids, attention_mask=torch.ones_like(input_ids))
            predictions = outputs[0, -1, :]
            predicted_index = torch.argmax(predictions).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

            if predicted_token == '[SEP]':
                break

            input_ids = torch.cat([input_ids, torch.tensor([[predicted_index]]).to(device)], dim=1)

    return tokenizer.decode(input_ids[0])


if __name__ == "__main__":
    main()
    # 加载训练好的模型并生成文本
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # vocab_size = len(tokenizer.vocab)
    # model = LanguageModel(vocab_size)
    # model.load_state_dict(torch.load("bert_language_model.pth"))
    # start_text = "李慕的意识在消失"
    # generated_text = generate_text(model, tokenizer, start_text)
    # print(f"输入: {start_text}")
    # print(f"生成: {generated_text}")