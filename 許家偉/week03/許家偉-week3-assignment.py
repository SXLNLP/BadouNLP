#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
生成隨機字符串，確保每個字符串都包含字符'a'
使用RNN神經網路進行多分類任務
分類目標：預測字符'a'在字符串中第一次出現的位置索引

"""

class RNNClassifier(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding層
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # RNN層，輸出的形狀是(batch_size, sentence_length, vector_dim)
        self.classify = nn.Linear(vector_dim, sentence_length)  # 線性層，輸出大小為句子長度（多分類）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵損失函數

    def forward(self, x, y=None): # x shape: (batch_size, sentence_length)
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sentence_length, vector_dim)
        # 取最後一個時間步的輸出
        last_hidden = rnn_out[:, -1, :]  # (batch_size, vector_dim)
        y_pred = self.classify(last_hidden)  # (batch_size, sentence_length)
        if y is not None:
            return self.loss(y_pred, y)  # 計算損失
        else:
            return y_pred  # 返回預測結果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "a你好我們在測試defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("a") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # 生成隨機字符串
        chars = [random.choice(list(vocab.keys())[1:]) for _ in range(sentence_length)]
        # 確保字符串中包含'a'
        if 'a' not in chars:
            pos = random.randint(0, sentence_length-1)
            chars[pos] = 'a'
        target_pos = chars.index('a') # 找到'a'的位置
        x = [vocab.get(char, vocab['unk']) for char in chars] # 將字符轉換為索引
        dataset_x.append(x)
        dataset_y.append(target_pos)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = RNNClassifier(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   # 建立200個用於測試的樣本
    print("本次預測集中共有%d個樣本" % len(y))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)      # 模型預測
        pred_positions = torch.argmax(y_pred, dim=1) # 獲取預測的最大概率位置
        # 計算正確預測的數量
        for pred, true in zip(pred_positions, y):
            if pred == true:
                correct += 1
            else:
                wrong += 1
    print("正確預測個數：%d, 正確率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置參數
    epoch_num = 20        # 訓練輪數
    batch_size = 32       # 每次訓練樣本個數
    train_sample = 1000   # 每輪訓練總共訓練的樣本總數
    char_dim = 64         # 每個字的維度
    sentence_length = 10  # 樣本文本長度
    learning_rate = 0.001 # 學習率
    
    vocab = build_vocab()
    print("詞彙表大小：", len(vocab))
    model = build_model(vocab, char_dim, sentence_length)
    print("模型結構：")
    print(model)
    
    # 選擇優化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 記錄訓練過程
    train_losses = []
    train_accs = []
    
    # 訓練過程
    for epoch in range(epoch_num):
        model.train()
        epoch_losses = []
        
        for epoch in range(int(train_sample / batch_size)):
            # 構造一組訓練樣本
            x, y = build_dataset(batch_size, vocab, sentence_length)
            
            # 前向傳播和反向傳播
            optim.zero_grad()    # 梯度歸零
            loss = model(x, y)   # 計算 loss
            loss.backward()      # 計算梯度
            optim.step()         # 更新權重
            
            epoch_losses.append(loss.item())
        
        # 計算平均損失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 評估模型
        model.eval()
        acc = evaluate(model, vocab, sentence_length)
        train_accs.append(acc)
        
        # 打印訓練信息
        print(f"Epoch [{epoch+1}/{epoch_num}]")
        print(f"平均損失: {avg_loss:.4f}")
        print(f"準確率: {acc:.4f}")
        print("="*50)
    
    # 保存模型
    torch.save(model.state_dict(), "RNN-model.pth")
    print("模型已保存到 RNN-model.pth")
    
    # 保存詞表
    with open("RNN-vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("詞表已保存到 RNN-vocab.json")
    
    # 繪製訓練過程圖表
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        # 繪製損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 繪製準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("訓練曲線已保存到 training_curves.png")
    except ImportError:
        print("未安裝 matplotlib，跳過繪製訓練曲線")
    
    return train_losses, train_accs

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    # 加載模型參數
    char_dim = 64  # 每個字的維度
    sentence_length = 10  # 樣本文本長度
    
    # 加載詞表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    print("詞彙表大小：", len(vocab))
    
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 測試模式
    
    # 處理輸入字符串
    results = []
    for input_string in input_strings:
        # 確保字符串長度符合要求
        if len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]
        elif len(input_string) < sentence_length:
            input_string = input_string + 'pad' * (sentence_length - len(input_string))
        
        # 將字符轉換為索引
        x = [vocab.get(char, vocab['unk']) for char in input_string]
        x = torch.LongTensor([x])  # 添加批次維度
        
        # 模型預測
        with torch.no_grad():
            y_pred = model(x)
            pred_position = torch.argmax(y_pred, dim=1).item()
            probabilities = torch.softmax(y_pred, dim=1)[0]
            
            top3_probs, top3_positions = torch.topk(probabilities, 2) # 獲取前兩個最可能的位置
            
            result = {
                'input': input_string,
                'predicted_position': pred_position,
                'top3_predictions': [
                    {'position': pos.item(), 'probability': prob.item()}
                    for pos, prob in zip(top3_positions, top3_probs)
                ]
            }
            results.append(result)
    
    #打印預測結果
    print("\n預測結果：")
    print("="*50)
    for result in results:
        print(f"輸入字符串：{result['input']}")
        print(f"預測位置：{result['predicted_position']}")
        print("前兩個最可能的位置：")
        for pred in result['top3_predictions']:
            print(f"  位置 {pred['position']}: {pred['probability']:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    # 訓練模型
    main()
    
    # 測試預測
    test_strings = ["bbaf12d4sh","cfesoja190","a4527byyed"]
    predict("RNN-model.pth", "RNN-vocab.json", test_strings)
