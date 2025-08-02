import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from transformers import BertTokenizer, BertModel

"""
基于BERT的问答生成模型(SFT)
输入新闻标题，生成对应的新闻内容
"""


class QALanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(QALanguageModel, self).__init__()
        # 使用预训练的BERT模型作为编码器
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        # 线性层，用于预测下一个token
        self.classify = nn.Linear(hidden_size, vocab_size)
        # 损失函数
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，实现因果注意力机制
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载数据
def load_qa_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append({
                    'question': item['title'], 'answer': item['content']
                })
    return data


# 构建样本
def build_qa_sample(tokenizer, data_item, max_length=128):
    question = data_item['question']
    answer = data_item['answer']

    # 编码问题部分
    question_tokens = tokenizer.encode(
        question,
        add_special_tokens=True,  # 添加[CLS]和[SEP]
        truncation=True,
        max_length=max_length // 2
    )

    # 编码答案部分
    answer_tokens = tokenizer.encode(
        answer,
        add_special_tokens=False,  # 不添加特殊token，要连接到问题后
        truncation=True,
        max_length=max_length // 2
    )

    # 组合输入序列: [CLS] question [SEP] answer [SEP]
    input_tokens = question_tokens + answer_tokens + [tokenizer.sep_token_id]

    # 截断到最大长度
    if len(input_tokens) > max_length:
        input_tokens = input_tokens[:max_length]

    # 构建标签序列，padding为0(不参与计算)，答案部分保留真实token
    labels = [0] * len(question_tokens) + answer_tokens + [tokenizer.sep_token_id]
    if len(labels) > max_length:
        labels = labels[:max_length]

    # padding
    while len(input_tokens) < max_length:
        input_tokens.append(tokenizer.pad_token_id)

    while len(labels) < max_length:
        labels.append(0)  # 0作为忽略的标签ID

    return input_tokens, labels


# 建立数据集
def build_qa_dataset(sample_length, tokenizer, data, max_length=128):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # 随机选择一个数据项
        item = random.choice(data)
        x, y = build_qa_sample(tokenizer, item, max_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(pretrain_model_path):
    model = QALanguageModel(768, 21128, pretrain_model_path)
    return model


# 根据问题生成答案
def generate_answer(question, model, tokenizer, max_length=128, max_answer_length=80):
    model.eval()
    question_size = len(question)
    with torch.no_grad():
        pred_char = ''
        # 逐步生成答案
        for _ in range(max_answer_length):
            question += pred_char
            question_tokens = tokenizer.encode(
                question,
                add_special_tokens=True,
                return_tensors='pt'
            )

            if torch.cuda.is_available():
                question_tokens = question_tokens.cuda()

            # 如果达到最大长度，停止生成
            if question_tokens.shape[1] >= max_length:
                break

            # 获取模型预测
            y_pred = model(question_tokens)
            # 获取最后一个token的预测分布
            next_token_logits = y_pred[0][-1]
            # 选择概率最高的token
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            pred_char = ''.join(tokenizer.decode(next_token_id))

        return question[question_size:]


def train(data_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    max_length = 128  # 序列最大长度
    learning_rate = 0.0001  # 学习率 (比之前更小，因为是微调)

    # 预训练模型路径
    pretrain_model_path = r'D:\python\resources\workspace\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 加载问答数据
    data = load_qa_data(data_path)
    print(f'加载了 {len(data)} 条问答数据')

    # 建立模型
    model = build_model(pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # 建立优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('模型加载完毕，开始训练')

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 构建一批训练样本
            x, y = build_qa_dataset(batch_size, tokenizer, data, max_length)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print(f'=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss)}')

        # 测试生成效果
        test_items = random.sample(data, 5)  # 随机选择3个测试样本
        for item in test_items:
            question = item['question']
            true_answer = item['answer']
            generated_answer = generate_answer(question, model, tokenizer, max_length)
            print(f'问题: {question}')
            print(f'真实答案: {true_answer}')
            print(f'生成答案: {generated_answer}\n')

    if save_weight:
        model_path = os.path.join('model', 'bert_sft_qa_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f'保存模型至 {model_path}')


if __name__ == '__main__':
    train('./sample_data.json', False)
