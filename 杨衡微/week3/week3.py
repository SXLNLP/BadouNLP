# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

使用embedding，rnn， crossentropy来判断
一句话中 'a'首次出现的index
如果'a'没有出现，则告知不存在。
假如一句话有10个字，那么crossentropy要输出11个结果，最后一个结果是'a'不存在的概率

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, rnn_hidden_size=128):
        super(TorchModel, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # rnn层输入形状：(batch_size, seq_len, input_size)
        # 输出形状：(batch_size, seq_len, hidden_size)
        self.rnn = nn.RNN(vector_dim, rnn_hidden_size, batch_first=True)
        # linear层输出维度 = sentence_length + 1 (多了一个"不存在"类)
        self.classify = nn.Linear(rnn_hidden_size, sentence_length + 1)
        self.loss = nn.CrossEntropyLoss()
        self.sentence_length = sentence_length

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # embedding层为input添加了一个维度
        # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.embedding(x)
        # rnn层(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, rnn_hidden_size)
        output, hidden = self.rnn(x)  # 需要解包
        # 将rnn层的output转变一下，取每一句话的最后一个字的输出
        # 也就是一句话只有一个输出
        # (batch_size, sen_len, rnn_hidden_size) -> (batch_size, rnn_hidden_size)
        x = output[:, -1, :]
        # (batch_size, rnn_hidden_size) -> (batch_size, sentence_length+1)
        y_pred = self.classify(x)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出预测结果 (batch_size, sentence_length+1)


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# padding 永远是 index 0
# unkown字符 永远是 最后一个 index
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
# pad和unk加上方括号，如果没有方括号，可能出现问题：
# text = "I don't know, unk to me"  # 文本中真的有"unk"这个词
# 有方括号就很清楚：
# vocab = {"[pad]": 0, "[unk]": 31, "u": 1, "n": 2, "k": 3}
# 不会把文本中的"unk"误认为是特殊标记
def build_vocab():
    chars = "你我他adefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"[pad]": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    last_key_index = len(vocab)
    vocab["[unk]"] = last_key_index
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# x1 = ['a', 'b', 'c', 'a', 'd']  -> y1 = 0 (第一个'a'在位置0)
# x1 = ['b', 'a', 'c', 'd', 'e']  -> y1 = 1 (第一个'a'在位置1)
# x1 = ['b', 'c', 'd', 'e', 'c']  -> y1 = sentence_length = 6 ('a'不存在)
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    # x = ['a', 'c', 'd'..., 'e'], 长度为sentence_length
    # x_index = [0, 1, 2, 0, 3...], 长度为sentence_length
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x_index = [vocab.get(char, vocab["[unk]"]) for char in x]

    # 找'a'的第一次出现位置
    try:
        y = x.index("a")  # 找到'a'的第一个位置
    except ValueError:
        y = sentence_length  # 'a'不存在，标签为sentence_length

    return x_index, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(batch_size, vocab, sentence_length):
    # 假设batch_size = 3, sentence_length = 5
    # 这意味着 dataset_x 里面有 3个sample，也就是3句话，每句话5个字
    # dataset_x[ith_sample] 可以取出任一句话
    # dataset_x[ith_sample][jth_word] 可以取出某句话里的 特定一个字。
    dataset_x = [] #(batch_size, sentence_length) 也就是 (batch_size, sen_len) 
    dataset_y = [] #（batch_size, ) 1 dimension，储存对每个sample，每句话的预测值（标量）
    for _ in range(batch_size):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)  # x 是 [0, 1, 2,....]，shape （sentence_length, )，代表一句话里每个字对应的index
        dataset_y.append(y)  # y是0, 代表这句话里，这个sample里'a'在第一个位置。是一个scalar 标量
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vector_dim, sentence_length, vocab, rnn_hidden_size=128):
    model = TorchModel(vector_dim, sentence_length, vocab, rnn_hidden_size)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    # x 的shape： （batch_size, sentence_length+1）
    # y 的shape： （batch_size, ),每一个sample的预测值
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # (200, sentence_length+1)
        predicted_positions = torch.argmax(y_pred, dim=-1)  # (200,)，对最后一维进行argmax
        correct = (predicted_positions == y).sum().item()
    return correct / len(y)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    
    # 建立字表
    vocab = build_vocab()
    print(f"词汇表大小: {len(vocab)}")
    
    # 建立模型 - 修正参数顺序
    model = build_model(char_dim, sentence_length, vocab)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        # 计算每轮的批次数
        num_batches = train_sample // batch_size
        
        for batch in range(num_batches):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss: {avg_loss:.6f}")
        
        # 测试本轮模型结果
        acc = evaluate(model, vocab, sentence_length)
        print(f"准确率: {acc:.4f}")
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("模型已保存到 model.pth")
    
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)
    print("词汇表已保存到 vocab.json")
    
    # 打印训练日志
    print("\n训练总结:")
    for i, (acc, loss) in enumerate(log):
        print(f"Epoch {i+1}: 准确率={acc:.4f}, 损失={loss:.6f}")
    
    return log

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    
    # 加载词汇表
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f) # ? block
    
    # 建立模型 - 修正参数顺序
    model = build_model(char_dim, sentence_length, vocab)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # 预处理输入数据
    x = []
    for input_string in input_strings:
        # 截断或填充到指定长度
        if len(input_string) > sentence_length:
            processed_string = input_string[:sentence_length]
        else:
            processed_string = input_string + '[pad]' * (sentence_length - len(input_string))
        
        # 转换为索引
        x_indices = [vocab.get(char, vocab["[unk]"]) for char in processed_string]
        x.append(x_indices)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 获取概率分布
        predicted_positions = torch.argmax(result, dim=-1)  # 获取预测位置
    
    # 打印结果
    for i, input_string in enumerate(input_strings):
        pred_pos = predicted_positions[i].item()
        max_prob = torch.max(result[i]).item()
        
        if pred_pos == sentence_length:
            position_desc = "不存在"
        else:
            position_desc = f"位置{pred_pos}"
        
        print(f"输入：'{input_string}' -> 预测'a'首次出现在：{position_desc}, 置信度：{max_prob:.4f}")
        
        # 显示完整概率分布（可选）
        print(f"  概率分布：{result[i].numpy()}")
        print()

if __name__ == "__main__":
    test_strings = ["abcdef", "badefg", "你我aekn","xyzabr", "ger讲as","--zgha","mnopqr"]
    main()
    predict("model.pth", "vocab.json", test_strings)
