import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# 模型定义
# class PositionPredictor(nn.Module):
#     def __init__(self, vocab_size, vector_dim, sentence_length):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, vector_dim)
#         self.linear = nn.Linear(vector_dim, sentence_length)

#     # def forward(self, x):
#     #     embeddings = self.embedding(x)  # (batch, seq_len, vector_dim)
#     #     summed = torch.sum(embeddings, dim=1)  # (batch, vector_dim)
#     #     out = self.linear(summed)  # (batch, sentence_length)
#     #     return out

#     def forward(self, x):
#         embeddings = self.embedding(x)  # (batch, seq_len, vector_dim)
#         output, hidden = self.rnn(embeddings)  # output: (batch, seq_len, hidden_dim), hidden: (1, batch, hidden_dim)
#         out = self.linear(hidden.squeeze(0))  # hidden.squeeze(0): (batch, hidden_dim)
#         return out


class PositionPredictor(nn.Module):
    def __init__(self, vocab_size, vector_dim, sentence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(128, sentence_length)

    def forward(self, x):
        embeddings = self.embedding(x)  # (batch, seq_len, vector_dim)
        output, hidden = self.rnn(embeddings)  # output: (batch, seq_len, hidden_dim)
        out = self.linear(hidden.squeeze(0))  # (batch, sentence_length)
        return out


# 构建模型
def build_model(vocab, char_dim, sentence_length):
    vocab_size = len(vocab)
    model = PositionPredictor(vocab_size, char_dim, sentence_length)
    return model

# 随机生成训练样本
def generate_sample(vocab_list, sentence_length):
    position = random.randint(0, sentence_length - 1)
    sentence = [random.choice(vocab_list) for _ in range(sentence_length)]
    sentence[position] = 'a'
    return ''.join(sentence), position

# 训练函数
def main():
    # 参数设置
    char_dim = 20
    sentence_length = 6
    vocab_list = list("abcdefghijklmnopqrstuvwxyz")
    vocab = {ch: i for i, ch in enumerate(vocab_list)}
    vocab["unk"] = len(vocab)
    
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, indent=2)

    model = build_model(vocab, char_dim, sentence_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for step in range(1000):
        batch_size = 16
        inputs, labels = [], []
        for _ in range(batch_size):
            sentence, pos = generate_sample(vocab_list, sentence_length)
            ids = [vocab.get(ch, vocab["unk"]) for ch in sentence]
            inputs.append(ids)
            labels.append(pos)
        inputs = torch.LongTensor(inputs)
        labels = torch.LongTensor(labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("模型训练完毕并已保存为 model.pth")

# 预测函数
def predict(model_path, vocab_path, input_strings):
    """
    使用训练好的模型预测字符 'a' 在输入句子中的位置。

    :param model_path: 模型参数文件路径
    :param vocab_path: 字符表 JSON 文件路径
    :param input_strings: 待预测的字符串列表
    """
    char_dim = 20
    sentence_length = 6

    # 加载词表和模型
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 构造输入张量
    input_tensor = []
    for s in input_strings:
        if len(s) != sentence_length:
            raise ValueError(f"输入句子长度必须为 {sentence_length}，但收到长度为 {len(s)} 的句子：{s}")
        ids = [vocab.get(char, vocab["unk"]) for char in s]
        input_tensor.append(ids)

    input_tensor = torch.LongTensor(input_tensor)

    # 模型预测
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=1)

    # 输出结果
    for i, s in enumerate(input_strings):
        print(f"输入：{s}，预测字符 'a' 的位置：{predictions[i].item()}，得分分布：{output[i].tolist()}")

# 主入口
if __name__ == "__main__":
    if not os.path.exists("model.pth"):
        main()

    # 示例预测
    test_sentences = [
        "bcdaaa",  # 多个 a
        "dabccd",
        "cccbab",
        "abcabc",
        "abccba"
    ]
    predict("model.pth", "vocab.json", test_sentences)
