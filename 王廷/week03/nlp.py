#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64, num_classes=7):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.GRU(vector_dim, hidden_size, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(hidden_size * 2, num_classes)     
        self.sentence_length = sentence_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size*2)
        y_pred = self.classify(output)  # (batch_size, sen_len, num_classes)
        y_pred = y_pred[:, -1, :]  # 取最后一个时间步的输出 (batch_size, num_classes)
        
        if y is not None:
            loss = nn.functional.cross_entropy(y_pred, y.long()) 
            return loss
        else:
            return y_pred  # 输出预测结果

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab

def build_sample(vocab, sentence_length):
    # 创建包含随机字母的字符串
    chars = list(vocab.keys())
    chars.remove("pad")
    chars.remove("unk")
    x = [random.choice(chars) for _ in range(sentence_length)]
    
    # 70%的概率包含'a'
    if random.random() < 0.7:
        # 随机选择插入位置 (0到sentence_length-1)
        pos = random.randint(0, sentence_length - 1)
        x[pos] = 'a'
        y = pos
    else:
        y = sentence_length  # 没有'a'的情况
    
    # 将字转换成序号
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 直接添加整数，不是列表
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length, num_classes=7):
    model = TorchModel(char_dim, sentence_length, vocab, num_classes=num_classes)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    
    # 统计各个位置的数量
    position_counts = {i: 0 for i in range(7)}
    for label in y.numpy():
        if label < 6:
            position_counts[label] += 1
        else:
            position_counts[6] += 1
    
    print("位置分布: ", position_counts)
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct = (predicted_classes == y).sum().item()
    
    accuracy = correct / len(y)
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率
    num_classes = 7  # 位置0-5 + 未出现(6)
    
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    accuracies = []
    losses = []
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        epoch_losses = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 评估
        acc = evaluate(model, vocab, sentence_length)
        accuracies.append(acc)
        
        print(f"=========\n第{epoch+1}轮平均loss: {avg_loss:.4f}, 准确率: {acc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
     
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    num_classes = 7
    
    # 加载字符表
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    
    # 反转vocab用于输出
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    model.load_state_dict(torch.load(model_path,weights_only = True))  # 加载训练好的权重
    
    # 处理输入
    x = []
    processed_strings = []
    for s in input_strings:
        # 截断或填充
        s = s[:sentence_length].ljust(sentence_length, ' ')
        ids = [vocab.get(c, vocab["unk"]) for c in s]
        x.append(ids)
        processed_strings.append(s)
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.LongTensor(x))  # 模型预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    print("\n预测结果:")
    print("=" * 50)
    for i, s in enumerate(processed_strings):
        position = preds[i].item()
        prob = probs[i][position].item()
        
        if position == 6:
            result = "未出现"
            # 高亮显示字符串中的'a'
            highlighted = ''.join([f"[{c}]" if c == 'a' else c for c in s])
        else:
            result = f"位置 {position}"
            # 高亮显示预测位置和所有'a'
            highlighted = []
            for j, c in enumerate(s):
                if j == position:
                    highlighted.append(f"<{c}>")
                elif c == 'a':
                    highlighted.append(f"[{c}]")
                else:
                    highlighted.append(c)
            highlighted = ''.join(highlighted)
        
        print(f"输入: {highlighted}")
        print(f"预测: {result} (概率: {prob:.4f})")
        print(f"概率分布: {[f'{p:.4f}' for p in probs[i].tolist()]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
    
    # 测试字符串
    test_strings = [
        "andgdgc", 
        "habgdg", 
        "uiuhu", 
        "nuh",
        "hugyeag",
        "bgdya",
        "audwge",
        "kiwi",
        "orange",
        "nothing",
        "aaaaaa",
        "a"
    ]
    
    predict("model.pth", "vocab.json", test_strings)
