#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn.functional as F

"""
用Bert + mask 替换lstm 完成自回歸語言生成任務
"""

# 簡化版本的BERT模型 - 避免配置問題
class SimpleBertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=8):
        super(SimpleBertModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 位置編碼
        self.position_embedding = nn.Embedding(512, hidden_size)
        
        # 多層Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 語言模型頭部
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 損失函數
        self.loss = nn.functional.cross_entropy
        
        # Mask token ID
        self.mask_token_id = vocab_size - 1
    
    def forward(self, x, y=None, mask_positions=None):
        batch_size, seq_len = x.shape
        
        # 創建位置編碼
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 詞嵌入 + 位置編碼
        x_emb = self.embedding(x) + self.position_embedding(positions)
        x_emb = self.dropout(x_emb)
        
        # Transformer編碼
        hidden_states = self.transformer(x_emb)
        
        # 語言模型頭部
        logits = self.lm_head(hidden_states)
        
        if y is not None:
            # 訓練模式：計算損失
            if mask_positions is not None:
                # 只計算mask位置的損失
                loss = 0
                for i, pos in enumerate(mask_positions):
                    if isinstance(pos, list):
                        pos = pos[0] if pos else 0
                    if pos < seq_len:
                        # 修復維度問題：確保目標是正確的形狀
                        target_token = y[i, pos].unsqueeze(0)  # [1]
                        pred_logits = logits[i, pos].unsqueeze(0)  # [1, vocab_size]
                        loss += self.loss(pred_logits, target_token)
                return loss / len(mask_positions) if len(mask_positions) > 0 else loss
            else:
                # 計算所有位置的損失
                return self.loss(logits.view(-1, self.vocab_size), y.view(-1))
        else:
            # 推理模式：返回概率分布
            return torch.softmax(logits, dim=-1)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=12):
        super(LanguageModel, self).__init__()
        
        # 保存重要參數
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 使用transformers的BertConfig
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=512,
            type_vocab_size=1,
            pad_token_id=0,
            mask_token_id=vocab_size - 1  # 使用最後一個token作為mask
        )
        
        # 創建BERT模型
        self.bert = BertModel(self.config)
        
        # 創建語言模型頭部（用於預測被mask的token）
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # 添加dropout防止過擬合
        self.dropout = nn.Dropout(0.1)
        
        # 定義損失函數
        self.loss = nn.functional.cross_entropy

    # 前向傳播方法 - 實現BERT+Mask的邏輯
    def forward(self, x, y=None, mask_positions=None):
        # x: 輸入序列 (batch_size, seq_len)
        # y: 目標序列 (batch_size, seq_len) 
        # mask_positions: mask位置列表
        
        batch_size, seq_len = x.shape
        
        # 步驟1: 創建attention mask（告訴BERT哪些位置需要關注）
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # 步驟2: 如果提供了mask位置，則在該位置進行mask
        if mask_positions is not None:
            for i, pos in enumerate(mask_positions):
                if isinstance(pos, list):
                    # 如果pos是列表，取第一個元素
                    pos = pos[0] if pos else 0
                if pos < seq_len:
                    x[i, pos] = self.config.mask_token_id  # 將指定位置替換為mask token
        
        # 步驟3: BERT前向傳播
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 步驟4: 獲取最後一層的隱藏狀態
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # 步驟5: 通過語言模型頭部預測詞彙
        logits = self.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)
        
        # 步驟6: 根據是否提供目標來決定返回損失還是預測結果
        if y is not None:
            # 訓練模式：計算損失
            if mask_positions is not None:
                # 只計算mask位置的損失 - BERT的標準做法
                loss = 0
                for i, pos in enumerate(mask_positions):
                    if pos < seq_len:
                        loss += self.loss(logits[i:i+1, pos:pos+1], y[i:i+1, pos:pos+1])
                return loss / len(mask_positions) if len(mask_positions) > 0 else loss
            else:
                # 計算所有位置的損失
                return self.loss(logits.view(-1, self.vocab_size), y.view(-1))
        else:
            # 推理模式：返回概率分布
            return torch.softmax(logits, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本 - 修改为BERT Mask语言模型
#从文本中截取随机窗口，随机mask一些位置进行预测
def build_sample(vocab, window_size, corpus, mask_ratio=0.15):
    # 從語料中隨機選擇一個窗口
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    
    # 將字符轉換為數字ID
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = x.copy()  # 目標序列與輸入序列相同
    
    # 隨機選擇一些位置進行mask
    mask_positions = []
    vocab_size = len(vocab)
    
    for i in range(len(x)):
        if random.random() < mask_ratio:  # 15%的概率進行mask
            mask_positions.append(i)
            y[i] = x[i]  # 目標是預測被mask的原始token
            x[i] = vocab.get("<MASK>", vocab_size - 1)  # 使用mask token替換
    
    return x, y, mask_positions

#建立数据集 - 修改为支持BERT Mask语言模型
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    mask_positions_list = []
    
    # 生成指定數量的樣本
    for i in range(sample_length):
        x, y, mask_positions = build_sample(vocab, window_size, corpus)
        
        # 在數據準備階段就應用mask
        if mask_positions:
            for pos in mask_positions:
                if pos < len(x):
                    x[pos] = vocab.get("<MASK>", len(vocab) - 1)
        
        dataset_x.append(x)
        dataset_y.append(y)
        # 確保mask_positions是整數列表
        if mask_positions:
            mask_positions_list.append(mask_positions[0] if len(mask_positions) > 0 else 0)
        else:
            mask_positions_list.append(0)
    
    # 轉換為tensor並返回
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), mask_positions_list

#建立模型 - 使用簡化版本的BERT模型
def build_model(vocab, hidden_size=256):
    # 獲取詞彙表大小
    vocab_size = len(vocab)
    
    # 創建簡化版本的BERT語言模型
    model = SimpleBertModel(vocab_size, hidden_size)
    
    return model

#文本生成测试代码 - 修改为BERT生成方式
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            
            # 取最後window_size個字符作為上下文
            current_text = openings[-window_size:]
            x = [vocab.get(char, vocab["<UNK>"]) for char in current_text]
            
            # 在最後一個位置添加mask token
            x.append(vocab.get("<MASK>", len(vocab) - 1))
            x = torch.LongTensor([x])
            
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 預測mask位置的token
            y = model(x)[0][-1]  # 取最後一個位置（mask位置）的預測
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    
    return openings

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


#计算文本ppl - 修改为BERT方式
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    
    with torch.no_grad():
        for i in range(1, len(sentence)):
            # 步驟1: 獲取上下文窗口
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            
            # 步驟2: 在最後添加mask token
            x.append(vocab.get("<MASK>", len(vocab) - 1))
            x = torch.LongTensor([x])
            
            # 步驟3: 獲取目標token
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 步驟4: 預測並計算概率
            pred_prob_distribute = model(x)[0][-1]  # 取mask位置的預測
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    hidden_size = 256     #簡化BERT隱藏層大小
    window_size = 10       #样本文本长度
    vocab = build_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    
    # 創建簡化BERT模型
    model = build_model(vocab, hidden_size)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 設置優化器（BERT需要較低的學習率）
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)   #建立优化器，降低學習率
    
    print("簡化BERT Mask語言模型加載完畢，開始訓練")
    
    # 步驟5: 開始訓練循環
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            # 步驟6: 構建訓練樣本
            x, y, mask_positions = build_dataset(batch_size, vocab, window_size, corpus)
            
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
                # mask_positions是列表，不需要轉移到GPU
            
            # 步驟7: 前向傳播和反向傳播
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask_positions)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

# 測試函數 - 驗證簡化BERT模型的基本功能
def test_bert_model():
    """
    測試簡化BERT模型的基本功能
    """
    print("開始測試簡化BERT模型...")
    
    # 創建一個簡單的詞彙表
    test_vocab = {"<pad>": 0, "<UNK>": 1, "<MASK>": 2, "a": 3, "b": 4, "c": 5}
    
    # 創建一個小型的簡化BERT模型
    model = SimpleBertModel(len(test_vocab), hidden_size=64, num_layers=2, num_heads=4)
    
    # 創建測試數據
    test_input = torch.LongTensor([[3, 4, 5, 2]])  # "abc[MASK]"
    test_target = torch.LongTensor([[3, 4, 5, 3]])  # 目標是預測"a"
    
    # 測試前向傳播
    try:
        output = model(test_input)
        print(f"✅ 前向傳播成功，輸出形狀: {output.shape}")
        
        # 測試損失計算
        loss = model(test_input, test_target, mask_positions=[3])
        print(f"✅ 損失計算成功，損失值: {loss.item():.4f}")
        
        print("🎉 簡化BERT模型測試通過！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 先運行測試確保模型正常工作
    #test_bert_model()
    
    # 如果測試通過，再運行訓練
    print("\n開始訓練模型...")
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", True)