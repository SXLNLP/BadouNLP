#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于pytorch的LSTM语言模型

 -》gpt2
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim) # 词表大小*嵌入维度
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True) # 输入特征的维度，隐藏状态的维度，LSTM 的层数，(batch_size, sequence_length, input_dim)
        self.bert = BertModel.from_pretrained("D:\worksapce\\ai_workspace\\nlp20\week6\\bert-base-chinese", return_dict=True)
        # 修改为bert预训练模型
        self.classify = nn.Linear(input_dim, vocab_size)
        # self.dropout = nn.Dropout(0.1)
        # self.loss = nn.functional.cross_entropy
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # pad位忽略

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        # y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        # if y is not None:
        #     return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        # else:
        #     return torch.softmax(y_pred, dim=-1)
    def forward(self, input_ids, attention_mask, y=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.classify(hidden_states)
        if y is not None:
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
            return loss
        return logits

#加载字表，其实就是建立一个字典，字符为键，索引为值，其中设置了<pad>字典值为0
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料，strip函数去掉行首空白字符和行尾换行符，最后返回连续的字符串语料进行训练
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    text = corpus[start:start + window_size]
    encode_result = tokenizer(text, padding="max_length", max_length=window_size + 2, truncation=True, return_tensors="pt")
    input_ids = encode_result["input_ids"][0]
    attention_mask = encode_result["attention_mask"][0]

    # 做 mask
    mask_token_id = tokenizer.mask_token_id
    labels = input_ids.clone()
    for i in range(1, len(input_ids) - 1):
        if random.random() < 0.15:
            input_ids[i] = mask_token_id
        else:
            labels[i] = -100  # 不是 mask 的不计算 loss
    return input_ids, labels, attention_mask

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    masks = []
    for _ in range(sample_length):
        x, y, mask = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        masks.append(mask)
    return torch.stack(dataset_x), torch.stack(dataset_y), torch.stack(masks)

#建立模型
def build_model(char_dim, vocab_size):
    return LanguageModel(char_dim, vocab_size)


#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        while not openings.endswith("\n") and len(openings) <= 30:
            ids = [tokenizer.cls_token_id] \
                  + tokenizer.convert_tokens_to_ids(list(openings[-window_size:])) \
                  + [tokenizer.mask_token_id] \
                  + [tokenizer.sep_token_id]
            attention_mask = [1] * len(ids)
            pad_len = window_size + 3 - len(ids)
            if pad_len > 0:
                ids += [0] * pad_len
                attention_mask += [0] * pad_len
            input_ids = torch.LongTensor([ids])#.cuda()
            attention_mask = torch.LongTensor([attention_mask])#.cuda()
            logits = model(input_ids, attention_mask)
            mask_idx = ids.index(tokenizer.mask_token_id)
            probs = torch.softmax(logits[0, mask_idx], dim=-1)
            next_id = sampling_strategy(probs)
            pred_char = tokenizer.convert_ids_to_tokens([next_id])[0]
            openings += pred_char if pred_char != "[SEP]" else "\n"
    return openings



def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        return int(torch.argmax(prob_distribution))
    else:
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

#计算文本ppl
def calc_perplexity(sentence, model, tokenizer, window_size):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    inputs = {k: v for k, v in inputs.items()}
    labels = inputs["input_ids"]
    with torch.no_grad():
        loss = model(inputs["input_ids"], inputs["attention_mask"], y=labels)
    return math.exp(loss.item())

def train(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    # char_dim = 256      #每个字的维度
    char_dim = 768         # 修改为bert的维度
    window_size = 10       #样本文本长度
    tokenizer = BertTokenizer.from_pretrained("D:\worksapce\\ai_workspace\\nlp20\week6\\bert-base-chinese")
    # vocab = build_vocab("vocab.txt")       #建立字表，返回的是一个字典
    vocab = tokenizer
    corpus = load_corpus(corpus_path)     #加载语料，返回的是一个字符串预料
    # model = build_model(vocab, char_dim)    #建立模型，传入字表和字维度
    model = build_model(char_dim, tokenizer.vocab_size)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            # if torch.cuda.is_available():
            #     x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, attention_mask=mask, y=y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        if not os.path.exists("model"):
            os.makedirs("model")
        torch.save(model.state_dict(), model_path)
        return

def main():
    # 1. 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained("D:\worksapce\\ai_workspace\\nlp20\week6\\bert-base-chinese")
    vocab_size = tokenizer.vocab_size

    # 2. 初始化模型
    model = LanguageModel(input_dim=768, vocab_size=vocab_size)

    # 3. 加载预训练权重（需要先训练模型保存权重）
    model_path = "D:\worksapce\\ai_workspace\\nlp20\week10\week10 文本生成问题\lstm语言模型生成文本\model\corpus.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"成功加载模型权重：{model_path}")
    else:
        print(f"警告：未找到模型权重文件{model_path}, 使用随机初始化的模型")

    # 4. 将模型设置为评估模式
    model.eval()

    # 5. 设置设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 6. 生成文本
    window_size = 10 # 与训练时长度一致
    starters = ["他缓缓走出来，看着那个","慢慢浮现出来"]
    for starter in starters:
        generated = generate_sentence(
            starter,
            model,
            tokenizer,
            window_size
        )
        print(f"起始: '{starter}'")
        print(f"生成: {generated}")
        print("-" * 80)

if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    # train("corpus.txt", True)
    main()

