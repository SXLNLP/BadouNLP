import numpy as np
import torch
import torch.nn as nn
import json
import random
import os
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, mask=None, y=None):
        if y is not None:
            print(mask.shape)
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


def load_corpus(path):
    corpus = []
    with open(path, encoding='gbk') as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus


def create_mask(s1, s2):
    len_s1 = s1 + 2
    len_s2 = s2 + 1
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    for i in range(len_s1):
        mask[i, len_s1:] = 0
    for i in range(len_s2):
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask


def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

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

def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        mask = create_mask(len(prompt_encode), len(answer_encode))
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def build_model(vocab,char_dim,pretrain_model_path):
    model = LanguageModel(768,21128,pretrain_model_path)
    return model

def sampling_strategy(prob_distribution):
    if random.random()>0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
    return np.random.choice(list(range(len(prob_distribution))),p=prob_distribution)

def generate_sentence(openings,model,tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <=30:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def main(corpus_path,save_weight=True):
    epoch_num = 20
    batch_size = 32
    char_dim = 768
    max_length =50
    vocab_size = 21128
    learning_rate = 0.001

    pretrain_model_path ="bert-base_chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus =load_corpus(corpus_path)
    train_data = build_dataset(tokenizer,corpus,max_length,batch_size)
    model = build_model(vocab_size,char_dim,pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(),lr =learning_rate)
    print("文本词表加载完毕,开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x,mask,y = x.cuda(),mask.cuda(),y.cuda()
            optim.zero_grad()
            loss = model(x,mask,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北美洲发现肥皂人",model,tokenizer))
        print(generate_sentence("朝鲜宣布将于4月发射人造卫星"),model,tokenizer)
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt","pth")
        model_path = os.path.join("model",base_name)
        torch.save(model.state_dict(),model_path)
        return


if __name__ =="__main__":
    main("sample_data.json",False)

