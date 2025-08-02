import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json
import gc

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, attention_mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=attention_mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            parsed_json = json.loads(line.strip())
            # line = re.sub(r"\s+", "", line)  #去掉空格
            corpus.append({"title": parsed_json["title"], "content": parsed_json["content"]})
    return corpus


#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []  #存储样本x
    dataset_y = []  #存储样本y
    attention_mask = torch.zeros((sample_length, window_size, window_size), dtype=torch.long)  #存储sample_length个mask
    # 随机生成sample_length个从0到len(corpus)的数
    indexs = np.random.randint(0, len(corpus), sample_length)
    #循环indexs，生成样本
    number = 0
    for index in indexs:
        title = corpus[index]["title"]
        content = corpus[index]["content"]
        x = tokenizer.encode(title,content, add_special_tokens=True, padding='max_length', truncation=True, max_length=window_size)   #将字转换成序号
        #向右移动一位，并用PAD填充
        y = x[1:] + [tokenizer.pad_token_id]  #将x向右移动一位，最后一个字用pad填充
        #创建mask
        mask = create_sentence_inverted_triangle_mask(x, tokenizer, max_length=window_size)
        dataset_x.append(x)  #将样本x添加到dataset_x
        dataset_y.append(y)  #将样本y添加到dataset_y
        attention_mask[number] = mask  #将mask添加到attention_mask
        number += 1
    #将dataset_x和dataset_y转换成tensor
    #注意：如果使用BertTokenizer，tokenizer.encode会返回一个list，所以需要转换成LongTensor
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), attention_mask

def create_sentence_inverted_triangle_mask(
    token_ids: list, 
    tokenizer, 
    max_length: int = 512
) -> torch.LongTensor:
    """
    为包含两句话的token序列创建mask，仅对第二句话应用倒三角mask，其余位置为1
    
    参数:
    token_ids: 包含两句话的token ID列表
    tokenizer: BERT分词器
    max_length: 最大序列长度
    
    返回:
    np.ndarray: 形状为[max_length, max_length]的mask矩阵
    """
    # 查找第一个[SEP]的位置（第一句话结束）
    try:
        first_sep_idx = token_ids.index(tokenizer.sep_token_id)
    except ValueError:
        # 如果没有找到[SEP]，将整个序列视为第一句话
        first_sep_idx = len(token_ids) - 1

    # 第二句话的起始位置（第一个[SEP]之后）
    second_sent_start = first_sep_idx + 1

    # 初始化全1矩阵（默认所有位置可见）
    mask = torch.ones((max_length, max_length), dtype=torch.long)

    # 对第二句话的token应用倒三角mask
    for i in range(second_sent_start, len(token_ids)):
        # 当前行对应第二句话的token
        # 设置当前行中第二句话token的可见性（倒三角）
        for j in range(second_sent_start, len(token_ids)):
            if j > i:  # 如果当前列在当前行之后（未来token）
                mask[i, j] = 0  # 屏蔽未来token

    # 确保padding位置被屏蔽
    seq_length = len(token_ids)
    mask[:, seq_length:] = 0  # 屏蔽超出序列长度的列
    mask[seq_length:, :] = 0  # 屏蔽超出序列长度的行
    return mask

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了[SEP]，或生成文本超过150字则终止迭代
        while pred_char != "[SEP]" and pred_char != "[PAD]" and len(openings) <= 150:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)



def train(corpus_path, save_weight=True):
    # 检查显存使用情况
    print(torch.cuda.memory_summary())
    gc.collect()  # 回收Python垃圾
    torch.cuda.empty_cache()  # 释放PyTorch缓存

    epoch_num = 20        #训练轮数
    batch_size = 80       #每次训练样本个数
    train_sample = 800   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 150       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率


    pretrain_model_path = r'/home/nbs07/model/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型

    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, attention_mask = build_dataset(batch_size, tokenizer, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, attention_mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("南宁否认限购松绑系救市：实施北部湾城镇体系同城化", model, tokenizer, window_size))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(r"汪桓玉\week11/sample_data.json", False)