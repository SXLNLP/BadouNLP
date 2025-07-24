# 第八周作業 - 三元組損失函數訓練

修改表示型文本匹配代碼，使用三元組損失函數訓練。

## 三元組損失函數實現

```python
def cosine_triplet_loss(self, a, p, n, margin=None):
    ap = self.cosine_distance(a, p)  # anchor 與 positive 的距離
    an = self.cosine_distance(a, n)  # anchor 與 negative 的距離
    if margin is None:
        # 使用配置文件中的邊距參數，如果沒有則默認為0.1
        margin_value = getattr(self, 'config', {}).get('triplet_margin', 0.1)
        diff = ap - an + margin_value
    else:
        diff = ap - an + margin.squeeze()
    return torch.mean(diff[diff.gt(0)])  # 只計算正值
```

## 三元組樣本生成

```python
def triplet_training_sample(self):
    standard_question_index = list(self.knwb.keys())
    
    # 隨機選擇一個標準問題作為錨點類別
    anchor_class = random.choice(standard_question_index)
    
    # 確保選中的類別至少有兩個樣本
    if len(self.knwb[anchor_class]) < 2:
        # 如果樣本不足，重新選擇
        return self.triplet_training_sample()
    
    # 從錨點類別中隨機選擇兩個樣本作為anchor和positive
    anchor, positive = random.sample(self.knwb[anchor_class], 2)
    
    # 隨機選擇一個不同的類別作為負樣本類別
    negative_classes = [cls for cls in standard_question_index if cls != anchor_class]
    if not negative_classes:
        # 如果沒有其他類別，重新選擇
        return self.triplet_training_sample()
    
    negative_class = random.choice(negative_classes)
    negative = random.choice(self.knwb[negative_class])
    
    return [anchor, positive, negative]
```

## 模型前向傳播

```python
def forward(self, sentence1, sentence2=None, sentence3=None, target=None):
    # 三元組訓練：傳入三個句子 (anchor, positive, negative)
    if sentence2 is not None and sentence3 is not None:
        anchor = self.sentence_encoder(sentence1)      # (batch_size, hidden_size)
        positive = self.sentence_encoder(sentence2)    # (batch_size, hidden_size)
        negative = self.sentence_encoder(sentence3)    # (batch_size, hidden_size)
        # 使用三元組損失函數
        return self.cosine_triplet_loss(anchor, positive, negative)
    # 孿生網絡訓練：傳入兩個句子
    elif sentence2 is not None:
        vector1 = self.sentence_encoder(sentence1)
        vector2 = self.sentence_encoder(sentence2)
        if target is not None:
            return self.loss(vector1, vector2, target.squeeze())
        else:
            return self.cosine_distance(vector1, vector2)
    # 單個句子編碼
    else:
        return self.sentence_encoder(sentence1)
```

## 訓練循環

```python
for index, batch_data in enumerate(train_data):
    optimizer.zero_grad()
    if cuda_flag:
        batch_data = [d.cuda() for d in batch_data]
    # 三元組訓練：anchor, positive, negative
    anchor, positive, negative = batch_data
    loss = model(anchor, positive, negative)
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()
```

## 損失函數原理

這個三元組損失函數確保：
- **anchor 與 positive 的距離小於 anchor 與 negative 的距離**
- **差距至少為 margin 值**
- **只有違反這個約束的樣本才會產生損失**

## 訓練流程

1. **數據生成**：每次從知識庫中選擇一個類別，從中選取兩個樣本作為 anchor 和 positive，從其他類別選取一個樣本作為 negative
2. **模型前向傳播**：將三個樣本輸入模型，得到三個向量表示
3. **損失計算**：使用三元組損失函數計算損失
4. **反向傳播**：更新模型參數

## 配置文件更新

```python
Config = {
    "model_path": "../model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path": "../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,
    "triplet_margin": 0.1,      # 三元組損失的邊距參數
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
```

## 優勢

這樣訓練出來的模型能夠：
- **更好地學習句子之間的語義相似性**
- **提高問答匹配的準確率**
- **增強模型的泛化能力**
- **減少過擬合現象**