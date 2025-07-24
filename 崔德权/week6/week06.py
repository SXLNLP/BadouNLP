#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, 3)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)
        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)+1
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set("abc") & set(x) and not set("xyz") & set(x):
        y = 0
    elif not set("abc") & set(x) and set("xyz") & set(x):
        y = 1
    else:
        y = 2
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    
    # 计算BERT参数量
    bert_params = sum(p.numel() for p in model.bert.parameters())
    print(f"BERT参数数量: {bert_params}")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")
    
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200
    x, y = build_dataset(total, vocab, sample_length)
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d"%(y.tolist().count(0), y.tolist().count(1), y.tolist().count(2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f"%(correct, total, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 15
    batch_size = 20
    train_sample = 1000
    char_dim = 768
    sentence_length = 6
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)  # 这里会打印参数量
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == "__main__":
    main()