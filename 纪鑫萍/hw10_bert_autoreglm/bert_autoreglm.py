#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer
from loader import DataGenerator
from torch.utils.data import DataLoader

"""
基于pytorch：使用Bert+mask做自回归语言模型训练
1、加载分词器（创建数据集和训练模型时使用）；设置是否添加特殊标记
2、初始化模型
__init__
（1）加载预训练Bert模型作为编码器
（2）加载线性层用于预测下一个token
（3）实现模型训练：
    获取BERT编码器的输出、获取最后一层的隐藏状态
    应用自回归掩码到注意力权重（可选，用于可视化）
    预测下一个token
    如果target为空，则输出预测值，否则计算loss 
3、准备训练数据
4、模型训练
（1）
4、生成文本测试

"""

BERT_PATH = r"/Users/juju/BaDou/bert-base-chinese"


class LanguageModel(nn.Module):
    def __init__(self, input_dim):
        super(LanguageModel, self).__init__()
        self.encoder = BertModel.from_pretrained(BERT_PATH, output_attentions=True, output_hidden_states=True, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.classify = nn.Linear(input_dim, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # todo
    def forward(self, x, attention_mask=None, y=None):
        #  获取BERT编码器的输出、获取最后一层的隐藏状态
        outputs = self.encoder(
            x,
            attention_mask=attention_mask,
            output_attentions=True,  # 输出注意力权重用于调试
            output_hidden_states=True
        )

        #  应用自回归掩码到注意力权重
        modified_attentions = []
        if outputs.attentions is not None:
            batch_size, num_heads, seq_len, _ = outputs.attentions[0].shape
            right_top_mask = torch.triu(torch.ones(seq_len, seq_len) * float(-1e9), diagonal=1)  # ❓
            # 将自回归掩码应用到每层的注意力权重
            for attention in outputs.attentions:
                masked_attention = attention + right_top_mask.unsqueeze(0).unsqueeze(0)
                masked_attention = nn.functional.softmax(masked_attention, dim=-1)
                modified_attentions.append(masked_attention)
            modified_attentions = torch.stack(modified_attentions)

        tuple_attentions = tuple(modified_attentions)
        outputs = self.encoder(x, attention_mask=attention_mask,
                               output_attentions=True,  # 输出注意力权重用于调试
                               output_hidden_states=True
                               )
        outputs.attentions = tuple_attentions
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        #  预测下一个token
        y_pred = self.classify(sequence_output)
        #  如果target为空，则输出预测值，否则计算loss
        if y is None:
            return torch.softmax(y_pred, dim=-1)
        return self.loss(y_pred.view(-1, self.tokenizer.vocab_size), y.view(-1))

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串

# todo
def build_dataset(data_path, tokenizer, max_length, batch_size, with_special_tokens):
    input_ids = []
    attention_mask = []
    targets_ids = []
    dg = DataGenerator(data_path, tokenizer, max_length, with_special_tokens)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=True)
    for index, data in enumerate(dl):
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        # 创建自回归训练的标签（右移一个位置）
        targets_ids = input_ids.clone()
        targets_ids[targets_ids == tokenizer.pad_token_id] = -100
        targets_ids = torch.roll(targets_ids, shifts=-1, dims=1)
        targets_ids[:, -1] = -100  # 最后一个位置没有下一个token，设为-100
    return input_ids, attention_mask, targets_ids


#文本生成测试代码
""" 
文本生成测试代码
参数：openings-引言/初始文本
1、使用模型预测
2、取出最后一个token的预测分布
3、应用top-k/top-t采样
4、
"""
# todo
@torch.no_grad()
def generate_sentence(openings, model, tokenizer, window_size, top_k, top_p):
    model.eval()
    input_ids = tokenizer.encode(openings, return_tensors='pt')
    for _ in range(window_size):
        # 获取预测结果
        output = model(input_ids)
        last_token = output[:, -1, :]

        # # 应用top-k或top-p采样
        # if top_k > 0:
        #     indices_to_remove = last_token < torch.topk(last_token, top_k)[0][..., -1, None]  # 取第一个返回值 获取最后一个元素，即第 K 大的值（排序后的最小值）
        #     last_token[indices_to_remove] = float('-1e9')
        # elif 0 < top_p < 1:
        #     # 保留累积概率超过top_p的最小token集合
        #     sorted_logits, sorted_indices = torch.sort(last_token, descending=True)
        #     cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        #     # 移除累积概率超过top_p的token
        #     sorted_indices_to_remove = cumulative_probs > top_p
        #     # 保留第一个超过top_p的token
        #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        #     sorted_indices_to_remove[..., 0] = 0
        #     # 应用掩码
        #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
        #     last_token.index_fill_(1, indices_to_remove, float('-1e9'))

        # 获取概率分布（采样下一个token）
        prob_distribution = nn.functional.softmax(last_token, dim=-1)
        next_token = torch.multinomial(prob_distribution, num_samples=1)  # 多项式采样（Multinomial Sampling）从概率分布中随机选择下一个 token

        # 添加预测的token
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 如果生成了结束标记，停止生成
        if (next_token == tokenizer.eos_token_id).all():
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)



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
    epoch_num = 20  #训练轮数
    batch_size = 64  #每次训练样本个数
    train_sample = 6400  #每轮训练总共训练的样本总数
    char_dim = 768  #每个字的维度
    window_size = 50  #样本文本长度
    data_path = "corpus.txt"
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    tokenizer.add_special_tokens({"eos_token": "[EOS]"})
    top_k = 5
    top_p = 0.9
    # vocab = build_vocab("vocab.txt")  #建立字表
    # corpus = load_corpus(corpus_path)  #加载语料
    model = LanguageModel(char_dim)  # todo
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)  # 建立优化器 todo
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # for batch in range(int(train_sample / batch_size)):
        # todo ✅
        input_ids, attention_mask, target_ids = build_dataset(data_path, tokenizer, window_size, batch_size, True)
        if torch.cuda.is_available():
            input_ids, attention_mask, target_ids = input_ids.cuda(), attention_mask.cuda(), target_ids.cuda()
        optim.zero_grad()  # 梯度归零
        # todo
        loss = model(input_ids, attention_mask, target_ids)  # 计算loss
        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        watch_loss.append(loss.item())  # todo
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size, top_k, top_p))  # todo
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size, top_k, top_p))  # todo
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
