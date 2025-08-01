# -*- coding: utf-8 -*-
import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
import math
from config import Config
#from evaluate import Evaluator
from loader import *
from model import LanguageModel
from transformers import BertModel, BertTokenizer

"""
基于pytorch的LSTM语言模型
"""


def main(config,save_weight=True):
    vocab_size = config["vocab_size"]  # 字表大小
    char_dim = config["char_dim"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    window_size = config["window_size"]
    train_sample = config["train_sample"]
    corpus_path = load_corpus(config["corpus_path"])     #加载语料
    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])  #加载Bert分词器

    #model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    model= LanguageModel(config)  #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")

    #训练
    for epoch in range(config["epoch_num"]):
        model.train()#训练模式
        watch_loss=[]
        for batch in range(int(train_sample / batch_size)):
            x,y=build_dataset(batch_size, tokenizer, window_size, corpus_path )#构建数据集
            if torch.cuda.is_available():
                x,y= x.cuda(), y.cuda()  #将数据放到GPU上
            optim.zero_grad()  #梯度清零
            loss = model(x, y)  #前向传播，计算loss
            loss.backward()  #反向传播
            optim.step()  #更新参数
            watch_loss.append(loss.item())  #记录loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
     # build_vocab_from_corpus("corpus/all.txt")
    main(Config)


