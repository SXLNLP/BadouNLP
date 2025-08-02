#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json

# 新增：引入transformers库
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BertTokenizer
from transformers import BertModel

"""
基于pytorch的LSTM语言模型
可选BERT作为encoder，支持causal mask实现自回归生成
"""

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, encoder_type='lstm', bert_name=(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese")):
        super(LanguageModel, self).__init__()
        print("BERT模型加载路径：", bert_name)
        self.encoder_type = encoder_type
        self.vocab = vocab
        if encoder_type == 'lstm':
            self.embedding = nn.Embedding(len(vocab), input_dim)
            self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
            self.classify = nn.Linear(input_dim, len(vocab))
            self.dropout = nn.Dropout(0.1)
        elif encoder_type == 'bert':
            self.bert_name = bert_name
            # 配置BERT支持decoder模式（causal mask）
            config = BertConfig.from_pretrained(bert_name)
            config.is_decoder = True  # 允许causal mask
            #attn_implementation="eager" 使用eager模式，可以输出注意力矩阵，查看mask是否生效
            self.bert = BertModel.from_pretrained(bert_name, config=config,attn_implementation="eager")
            self.tokenizer = BertTokenizer.from_pretrained(bert_name)
            # 使用BERT的词表大小，而不是传入的vocab大小
            self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if self.encoder_type == 'lstm':
            x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
            x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        elif self.encoder_type == 'bert':
            # 直接使用输入的token ID，不需要重新编码
            device = next(self.parameters()).device
            input_ids = x.to(device)            
            # 构造 causal mask（自回归掩码）
            batch_size, seq_len = input_ids.size()
            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            
            qatrillrightmask = torch.tril(torch.ones((119, 119)))
            qatrillleftmask = torch.ones((119,31))
            qatrillmask = torch.cat([ qatrillleftmask, qatrillrightmask], dim=1)
            topmask = torch.ones(( 31,150))
            topmask[:, 119:] = 0
            onemask = torch.cat([ topmask, qatrillmask], dim=0)
            mask = onemask.unsqueeze(0).repeat(batch_size, 1, 1) 
            #这里注意：只有训练才用mask，预测时不用mask
            #output_attentions=True 输出注意力矩阵,查看mask后的注意力矩阵，判断mask是否生效
            if y is not None:
                outputs = self.bert(input_ids=input_ids,attention_mask=mask,output_attentions=True)
            else:
                outputs = self.bert(input_ids=input_ids)
            x = outputs[0]  # tuple的第一个元素是last_hidden_state
            y_pred = self.classify(x)      # (batch, seq_len, vocab_size)
        if y is not None:
            # y_pred: 形状通常是 (batch_size, seq_len, vocab_size)
            # y:      形状通常是 (batch_size, seq_len)
            # y_pred.view(-1, y_pred.shape[-1]):
            #   - view(-1, y_pred.shape[-1])->view(-1, vocab_size)把前面所有维度展平成一维，最后一维保持vocab_size
            #   - 结果形状是 (batch_size * seq_len, vocab_size)
            #   - -1 表示自动推断（这里就是 batch_size * seq_len）
            # y.view(-1)：
            #   - 把y展平成一维
            #   - 结果形状是 (batch_size * seq_len,)
            # 这样可以把每个token的预测和标签都对齐，适配nn.CrossEntropyLoss的输入要求
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

def load_tatilcontext(path):
    title = []
    content = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title.append(line["title"])
            content.append(line["content"])
            
    return title,content
#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机文本训练样本（标准并行预测，BERT使用causal mask）
def build_sample_bert( window_size, corpus, tokenizer=None):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位，标准并行预测
    
    # 用BERT分词器的vocab.get方法，不添加pad
    x_tokens = [tokenizer.vocab.get(char, tokenizer.unk_token_id) for char in window]
    y_tokens = [tokenizer.vocab.get(char, tokenizer.unk_token_id) for char in target]
    return x_tokens, y_tokens
def build_dataset_AandQ(sample_length, window_size, tatil_maxlen, context_maxlen, tokenizer=None):
    t,c = load_tatilcontext(os.path.dirname(os.path.abspath(__file__))+"/sample_data.json")
    index = random.randint(0,len(t)-sample_length)
    dataset_x = []
    dataset_y = []
    ask = []
    for _ in range(sample_length):
        xt = tokenizer.encode(t[index], add_special_tokens=False, padding='max_length', truncation=True, max_length=tatil_maxlen)   #将字转换成序号
        xc1 = tokenizer.encode(c[index][0:], add_special_tokens=False, padding='max_length', truncation=True, max_length=context_maxlen-1)   #将字转换成序号
        xc2 = tokenizer.encode(c[index][1:], add_special_tokens=False, padding='max_length', truncation=True, max_length=context_maxlen-1)   #将字转换成序号

        x = xt +[tokenizer.sep_token_id]+ xc1 
        y = xt +[tokenizer.sep_token_id]+ xc2
        dataset_x.append(x)
        dataset_y.append(y)
        ask.append(t[index])
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y),ask

# 组装数据集（标准并行预测，BERT使用causal mask）
def build_dataset_bert(sample_length, window_size, corpus, tokenizer=None):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample_bert( window_size, corpus, tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 恢复LSTM分支的原始数据构造（并行预测序列）
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
# 新增encoder_type参数
def build_model(vocab, char_dim, encoder_type='lstm'):
    model = LanguageModel(char_dim, vocab, encoder_type=encoder_type)
    return model

#文本生成测试代码
#openings 引言 字符串
# 新增encoder_type参数
def generate_sentence(openings, model, vocab, window_size, encoder_type='lstm'):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 150:
            openings += pred_char #不断叠加预测的字符转位输入字符串
            if encoder_type == 'lstm':
                x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
                x = torch.LongTensor([x])
            elif encoder_type == 'bert':
                # BERT分支：使用tokenizer.vocab.get方法，类似数据生成
                x = [model.tokenizer.vocab.get(char, model.tokenizer.unk_token_id) for char in openings[-window_size:]]
                x = torch.LongTensor([x]) + [model.tokenizer.sep_token_id]
            
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x) #[1,window_size, len(vocab)]
            y = y[0][-1] #每次取最后一个字符的len(vocab)维度向量
            index = sampling_strategy(y)#将向量转为字符索引
            
            if encoder_type == 'bert':
                # BERT分支：使用BERT tokenizer解码
                pred_token = model.tokenizer.decode([index], skip_special_tokens=True)
                if pred_token:  # 如果不是空字符串
                    pred_char = pred_token
                else:
                    pred_char = ""
            else:
                pred_char = reverse_vocab[index]
    return openings

#采样策略
def sampling_strategy(prob_distribution):
    if random.random() > 0.5:
        strategy = "greedy" #贪婪采样greedy
    else:
        strategy = "sampling" #概率采样

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)



#训练流程
# 新增encoder_type参数
def train(corpus_path, save_weight=True, encoder_type='lstm'):
    print("train函数已进入", flush=True)
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 1280     #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    learning_rate = 0.001  #学习率lstm用0.01，bert用0.001，bert如果学习率太大，会效果很差，收敛慢。
    title_maxlen = 30
    context_maxlen = 120
    
    
    vocab = build_vocab(os.path.dirname(os.path.abspath(__file__))+"/vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    tokenizer = BertTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese")
    model = build_model(vocab, char_dim, encoder_type=encoder_type)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            if encoder_type == 'bert':
                x,y,ask = build_dataset_AandQ(batch_size, window_size, title_maxlen, context_maxlen, tokenizer)
                # x, y = build_dataset_bert(batch_size, window_size, corpus, tokenizer=model.tokenizer)
            else:
                x, y = build_dataset(batch_size, vocab, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # print("x shape:", x.shape, "y shape:", y.shape, flush=True)
            
            # 添加调试信息：检查训练数据
            # if encoder_type == 'bert':
            #     print(f"Batch {batch} - Training data:")
            #     print(f"  x shape: {x.shape}, y shape: {y.shape}")
            #     print(f"  Sample x[0]: {x[0].tolist()}")
            #     print(f"  Sample y[0]: {y[0].tolist()}")
            #     print(f"  Decoded x[0]: {model.tokenizer.decode(x[0].tolist(), skip_special_tokens=True)}")
            #     print(f"  Decoded y[0]: {model.tokenizer.decode(y[0].tolist(), skip_special_tokens=True)}")
            
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)), flush=True)
        x,y,ask = build_dataset_AandQ(2, window_size, title_maxlen, context_maxlen, tokenizer)
        print(generate_sentence(ask[0], model, vocab, window_size, encoder_type=encoder_type))
        print(generate_sentence(ask[1], model, vocab, window_size, encoder_type=encoder_type))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    print("主程序入口已进入", flush=True)
    # build_vocab_from_corpus("corpus/all.txt")
    # 传入encoder_type='bert'即可用BERT做encoder
    train(os.path.dirname(os.path.abspath(__file__))+"/corpus.txt", False, encoder_type='bert')
