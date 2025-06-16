#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'
        )
        self.classify = nn.Linear(hidden_size, 7) 
        self.loss = nn.CrossEntropyLoss()  

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)  # 输出为(batch_size, seq_len, hidden_size)
        output = output[:, -1, :]  # 取序列最后一个时间步的输出 (batch_size, hidden_size)
        y_pred = self.classify(output)  # (batch_size, 7)
        
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出概率分布

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 英文字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    # 随机生成字符串
    chars = list(vocab.keys())[1:-1]  # 排除pad和unk
    x = [random.choice(chars) for _ in range(sentence_length)]
    
    # 确定'a'第一次出现的位置
    try:
        position = x.index('a')
    except ValueError:
        position = 6  # 未出现标记为6
    
    # 将字符转换为索引
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, position

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    
    # 统计类别分布
    class_count = [0] * 7
    for label in y:
        class_count[label.item()] += 1
    print("各类别样本数量:", class_count)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 获取概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
        
        for pred, true in zip(predicted_classes, y):
            if pred == true:
                correct += 1
            else:
                wrong += 1
                
    accuracy = correct / (correct + wrong)
    print(f"正确预测: {correct}, 错误预测: {wrong}, 正确率: {accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 64
    train_sample = 3000
    char_dim = 32
    sentence_length = 6
    learning_rate = 0.001
    
    # 建立字表
    vocab = build_vocab()
    print("字符表大小:", len(vocab))
    
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        
        for _ in range(int(train_sample / batch_size)):
            # 构建训练批次
            x, y = build_dataset(batch_size, vocab, sentence_length)
            
            # 训练步骤
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / (train_sample / batch_size)
        print(f"Epoch {epoch+1}/{epoch_num}, Loss: {avg_loss:.4f}")
        
        # 每2轮评估一次
        if (epoch + 1) % 2 == 0:
            evaluate(model, vocab, sentence_length)
    
    # 保存模型和词汇表
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("rnn_vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 测试示例
    test_strings = [
        "bcdefg",  # 无a → 6
        "a12345",  # a在位置0
        "0a2345",  # a在位置1
        "01a345",  # a在位置2
        "012a45",  # a在位置3
        "0123a5",  # a在位置4
        "01234a",  # a在位置5
        "xayz12"   # a在位置1
    ]
    predict("rnn_model.pth", "rnn_vocab.json", test_strings)

def predict(model_path, vocab_path, input_strings):
    # 加载配置
    char_dim = 32
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    
    # 加载模型
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备输入数据
    input_indices = []
    for s in input_strings:
        # 截断或填充到固定长度
        s = s[:sentence_length].ljust(sentence_length, 'z')
        indices = [vocab.get(char, vocab['unk']) for char in s]
        input_indices.append(indices)
    
    # 转换为Tensor
    x = torch.LongTensor(input_indices)
    
    # 预测
    with torch.no_grad():
        probs = model(x)
        predictions = torch.argmax(probs, dim=1)
    
    # 打印结果
    position_names = ["位置0", "位置1", "位置2", "位置3", "位置4", "位置5", "未出现"]
    print("\n预测结果:")
    for i, s in enumerate(input_strings):
        pred_idx = predictions[i].item()
        prob = probs[i][pred_idx].item()
        print(f"输入: '{s}' → 预测: {position_names[pred_idx]} ({prob*100:.2f}%)")
        
        # 显示实际位置
        try:
            actual_pos = s.index('a')
        except ValueError:
            actual_pos = 6
        print(f"  实际: {position_names[actual_pos]}")

if __name__ == "__main__":
    main()
