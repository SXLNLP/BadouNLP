import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.nn.utils.rnn import pad_sequence

class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab_size, num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        self.rnn = nn.GRU(vector_dim, vector_dim, batch_first=True)  # GRU层替代池化层
        self.classify = nn.Linear(vector_dim, num_classes)  # 输出层对应11个类别(0-9位置+无a)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, lengths, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, vector_dim)
        
        # 打包变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)  # 获取最后一个隐藏状态
        hidden = hidden.squeeze(0)  # (1, batch_size, vector_dim) -> (batch_size, vector_dim)
        
        output = self.classify(hidden)  # (batch_size, num_classes)
        
        if y is not None:
            return self.loss(output, y)
        else:
            return torch.softmax(output, dim=-1)  # 返回概率分布

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 添加字符'a'
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, max_length=10):
    # 随机生成长度(1-10)的字符串
    length = random.randint(1, max_length)
    chars = random.choices(list(vocab.keys())[1:-1], k=length)  # 排除pad和unk
    
    # 查找第一个'a'的位置
    try:
        a_index = chars.index('a')
        label = a_index
    except ValueError:
        label = 10  # 没有'a'标记为10
        
    # 转换为索引并填充
    x = [vocab.get(c, vocab['unk']) for c in chars]
    return x, label, length

def build_dataset(sample_length, vocab):
    dataset_x = []
    dataset_y = []
    lengths = []
    for _ in range(sample_length):
        x, y, length = build_sample(vocab)
        dataset_x.append(torch.tensor(x, dtype=torch.long))
        dataset_y.append(y)
        lengths.append(length)
    
    # 填充序列
    dataset_x = pad_sequence(dataset_x, batch_first=True, padding_value=0)
    return dataset_x, torch.tensor(dataset_y, dtype=torch.long), torch.tensor(lengths)

def build_model(vocab, char_dim, num_classes):
    return TorchModel(char_dim, len(vocab), num_classes)

def evaluate(model, vocab):
    model.eval()
    x, y, lengths = build_dataset(200, vocab)
    correct = 0
    with torch.no_grad():
        outputs = model(x, lengths)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == y).sum().item()
    
    acc = correct / len(y)
    print(f"正确预测个数: {correct}, 正确率: {acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 32
    train_sample = 1000
    char_dim = 64
    num_classes = 11  # 0-9位置 + 无a(10)
    learning_rate = 0.001
    
    # 建立字表
    vocab = build_vocab()
    model = build_model(vocab, char_dim, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        
        # 分批训练
        for _ in range(0, train_sample, batch_size):
            x, y, lengths = build_dataset(min(batch_size, train_sample), vocab)
            optim.zero_grad()
            loss = model(x, lengths, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (train_sample / batch_size)
        print(f"Epoch {epoch+1}/{epoch_num}, Loss: {avg_loss:.4f}")
        evaluate(model, vocab)
    
    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def predict(model_path, vocab_path, input_strings):
    char_dim = 64
    num_classes = 11
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    
    model = TorchModel(char_dim, len(vocab), num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 处理输入
    sequences = []
    lengths = []
    for s in input_strings:
        seq = [vocab.get(c, vocab['unk']) for c in s]
        sequences.append(torch.tensor(seq, dtype=torch.long))
        lengths.append(len(seq))
    
    x = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    with torch.no_grad():
        outputs = model(x, torch.tensor(lengths))
        preds = torch.argmax(outputs, dim=1)
    
    for i, s in enumerate(input_strings):
        # 将类别10映射为-1
        result = preds[i].item()
        if result == 10:
            print(f"输入: {s}, 预测: 无a字符(-1), 概率分布: {outputs[i].numpy()}")
        else:
            print(f"输入: {s}, 预测: 位置{result}, 概率分布: {outputs[i].numpy()}")

if __name__ == "__main__":
    main()
    test_strings = ["apple", "banana", "cherry", "date", "elephant", "fox"]
    predict("rnn_model.pth", "vocab.json", test_strings)
