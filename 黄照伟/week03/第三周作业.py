import torch
import torch.nn as nn
import random
import json
import numpy as np




class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=20, num_layers=2, batch_first=True)  # 修改点1
        self.classifier = nn.Linear(20, sentence_length)  # 修改点2
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)  # [batch, 6, vector_dim]
        output, hidden = self.rnn(x)  # output形状: [batch, 6, 20]
        # 使用所有时间步的平均特征（替代仅最后一个时间步）
        x = output.mean(dim=1)  # [batch, 20]
        x = self.classifier(x)  # [batch, 6]
        if y is not None:
            return self.loss(x, y.long())
        else:
            return torch.softmax(x, dim=1)

#创建字符集
def build_vocab():
    chars = ""
    for i in range(ord('a'), ord('z')+1):
        chars += chr(i)
    vocab ={"pad":0}
    for index ,char in enumerate(chars):
        vocab[char] = index+1
    vocab["unk"] = len(vocab)
    return vocab

#随机生成样本 ,6位长度的字符串中包含字母a
def build_sample(vocab,sentence_length):
    #从字母选5个字符
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # chars = [random.choice(list(vocab.keys())) for i in range(sentence_length - 1)]
    chars = [random.choice(letters) for i in range(sentence_length - 1)]
    chars.append('a')
    random.shuffle(chars)
    x = ''.join(chars)
    #返回字符串 和 字母a的索引

    first_index =-1
    for i ,char in enumerate(x):
        if char == 'a':
            first_index = i
    #返回字符串和 下标
    return x,first_index

# 创建数据集
def build_dataset(sample_length, vocab, sentence_length):
    data_set_x = []
    data_set_y = []
    for i in range(sample_length):
        x_str, y = build_sample(vocab, sentence_length)
        x = [vocab.get(char, vocab["unk"]) for char in x_str]
        data_set_x.append(x)
        data_set_y.append(y)
    # 目标张量改为Long类型（类别索引）
    return torch.LongTensor(data_set_x), torch.LongTensor(data_set_y)
# 创建模型
def  build_model(vocab,char_dim,sentence_length):
    model = TorchModel(vector_dim=char_dim,sentence_length=sentence_length,vocab=vocab)
    return model

def evaluate(model,vocab,sample_length):
    model.eval()
    x,y =build_dataset(200,vocab,sample_length)
    print("本次预测集中共有%d个样本"%len(y))
    correct ,wrong =0,0
    with torch.no_grad():
        y_pred =model(x)
        for y_p ,y_t in zip(y_pred,y):
            if torch.argmax(y_p) == int(y_t):
                correct +=1
            else:
                wrong +=1
    # accuracy = correct/len(y)
    print("正确预测个数:%d,正确率:%f"%(correct,correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 20  # 增加训练轮次（由10→20）
    batch_size = 20
    train_sample = 1000
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005  # 增大学习率（由0.001→0.005）
    #建立字表
    vocab = build_vocab()
    model = build_model(vocab,char_dim,sentence_length)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log =[]
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x,y =build_dataset(train_sample,vocab,sentence_length)
            model.zero_grad()
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("-------------------- \n 第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab,train_sample)
        log.append([acc,np.mean(watch_loss)])
    #保存模型
    torch.save(model.state_dict(),"model2.pth")
    #保存词表
    writer = open("vocab2.json", "w", encoding="utf-8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#创建验证方法
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        # 修复：将填充字符改为单字符（此处用空格）
        processed_str = input_string[:sentence_length] if len(input_string) > sentence_length else input_string.ljust(sentence_length, ' ')  # 修改点：'pad'→' '
        x.append([vocab.get(char, vocab["unk"]) for char in processed_str])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    
    for i, input_string in enumerate(input_strings):
        pred_probs = result[i]
        pred_pos = torch.argmax(pred_probs).item()
        pred_prob = pred_probs[pred_pos].item()
        print("输入：%s, 预测a的位置：%d（概率：%.4f）" % (input_string, pred_pos, pred_prob))


if __name__ == "__main__":
    # main()
    test_string = [
        'moastb',
        'fawjvj',
        'orzoaf',
        'qftasg',
        'ajvvaa',
        'aonafo',
        'ajertz',
        'bxasek',
        'tpvaut',
        'iunagg',
        'gayevl',
        'vqczaj'
    ]
    predict('/Library/workerspace/python_test/badou_demo1/week03/model2.pth','/Library/workerspace/python_test/badou_demo1/week03/vocab2.json',test_string)

    # for i in range(6):
    #     letters = 'abcdefghijklmnopqrstuvwxyz'
    #     # chars = [random.choice(list(vocab.keys())) for i in range(sentence_length - 1)]
    #     chars = [random.choice(letters) for i in range(5)]
    #     chars.append('a')
    #     random.shuffle(chars)
    #     x = ''.join(chars)
    #     print(x)
