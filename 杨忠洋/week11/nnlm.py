#coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import logging
from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
基于pytorch的BERT语言模型
"""
config = {
    "bert_path": r"D:\AI\bert-base-chinese"
}


class LanguageModel(nn.Module):
    def __init__(self, weight, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.loss = nn.functional.cross_entropy
        self.weight = weight

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 构造前缀掩码
            mask = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
            # 整个输入序列可见
            mask[:, :, :24] = 1
            # 输出部分因果可见
            mask[:, 24:, 24:] = torch.tril(torch.ones(128, 128))
            if torch.cuda.is_available():
                mask = mask.cuda()
            bert_output, _ = self.bert(x, attention_mask=mask)
            # 计算loss时忽略输入部分
            y_pred = self.classify(bert_output)  #output shape:(batch_size, sen_len, hidden_size)
            return self.loss(input=y_pred[:, 24:, :].reshape(-1, y_pred.shape[-1]),
                             target=y.reshape(-1),
                             weight=self.weight,
                             ignore_index=0)
        else:
            bert_output, _ = self.bert(x)
            y_pred = self.classify(bert_output)
            return torch.softmax(y_pred, dim=-1)


#加载字表
def build_vocab():
    bert_vocab = BertTokenizer.from_pretrained(config["bert_path"]).vocab
    return bert_vocab


#加载语料
def load_data(path):
    questions = []
    answers = []
    with open(path, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            questions.append(line["title"])
            answers.append(line["content"])
    return questions, answers


def build_sample(vocab, question, answer):
    if len(question) > 24:
        question = question[:24]
    else:
        question += " " * (24 - len(question))
    if len(answer) > 128:
        answer = answer[:128]
    else:
        answer += " " * (128 - len(answer))
    input_text = question + answer
    target_text = input_text[len(question):]
    x = [vocab.get(char, vocab["[UNK]"]) for char in input_text]
    y = [vocab.get(char, vocab["[UNK]"]) for char in target_text]
    return x, y


#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
def build_dataset(questions, answers, vocab):
    dataset_x = []
    dataset_y = []
    for question, answer in zip(questions, answers):
        x, y = build_sample(vocab, question, answer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#建立模型
def build_model(vocab, weight):
    model = LanguageModel(weight, vocab)
    return model


#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
            try:
                pred_char.encode('gbk')
            except UnicodeEncodeError:
                pred_char = ""
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.7:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def build_weight(vocab, corpus):
    # 统计训练文本中出现的词
    word_freq = {}
    for char in corpus:
        word_freq[char] = word_freq.get(char, 0) + 1
    # 构建权重列表（默认值为 0）
    weights = torch.ones(len(vocab)) * 0.01  # 未出现的词权重较小
    for word, idx in vocab.items():
        if word in word_freq:
            weights[idx] = 1.0  # 出现的词权重为 1
    return weights


def train(corpus_path, save_weight=True):
    epoch_num = 1000  #训练轮数
    batch_size = 25  #每次训练样本个数
    window_size = 160  #样本文本长度
    vocab = build_vocab()  #建立字表
    questions, answers = load_data(corpus_path)
    cuda_flag = torch.cuda.is_available()
    weight = build_weight(vocab, questions + answers)
    if cuda_flag:
        weight = weight.cuda()
    model = build_model(vocab, weight)  #建立模型
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)  #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        logger.info("epoch %d begin" % epoch)
        model.train()
        watch_loss = []
        train_loss = []
        num_batches = int(len(questions) / batch_size)
        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = start_index + batch_size
            batch_questions = questions[start_index:end_index]
            batch_answers = answers[start_index:end_index]
            x, y = build_dataset(batch_questions, batch_answers, vocab)  #构建一组训练样本
            if cuda_flag:
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  #梯度归零
            loss = model(x, y)  #计算loss
            train_loss.append(float(loss))
            loss.backward()  #计算梯度
            optim.step()  #更新权重
            watch_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("潜伏者涂兆兴：敌人眼皮下掩护红色后代", model, vocab, window_size))
        print(generate_sentence("厦门公交车纵火案遇难者名单公布", model,
                                vocab, window_size))
    if not save_weight:
        return
    else:
        model_path = r"D:\AI\bert_nnlm_model.pth"
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
