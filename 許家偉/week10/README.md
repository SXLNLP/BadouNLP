# BERT+Mask 語言模型文本生成

## 概述
這個項目實現了一個基於簡化BERT+Mask語言模型的文本生成系統，將原本的LSTM語言模型替換為更先進的Transformer架構，同時避免了複雜的BERT配置問題。

## 主要修改

### 1. 模型架構變更
- **原模型**: LSTM語言模型
- **新模型**: 簡化BERT+Mask語言模型
- **主要組件**:
  - 詞嵌入層 (Word Embedding)
  - 位置編碼 (Position Embedding)
  - Transformer編碼器 (多層自注意力)
  - 語言模型頭部 (LM Head)

### 2. 訓練方式變更
- **原方式**: 序列到序列的預測
- **新方式**: Mask語言模型預訓練
  - 隨機mask輸入序列中的15%位置
  - 預測被mask位置的原始token
  - 使用雙向上下文信息
  - 在數據準備階段處理mask，避免inplace操作

### 3. 生成方式變更
- **原方式**: 自回歸生成
- **新方式**: 基於mask的生成
  - 在序列末尾添加mask token
  - 預測mask位置的token
  - 逐步生成文本

## 安裝依賴
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
# 運行測試確保模型正常工作
python bert_masked.py

# 測試函數會自動運行，然後開始訓練
```

## 主要特點
1. **簡化架構**: 使用PyTorch原生Transformer組件，避免複雜配置
2. **雙向上下文**: 利用Transformer的雙向注意力機制
3. **穩定訓練**: 解決了梯度計算中的inplace操作問題
4. **靈活的生成**: 支持多種採樣策略
5. **易於調試**: 代碼清晰，問題容易定位

## 參數配置
- `hidden_size`: 256 
- `num_layers`: 4 
- `num_heads`: 8 (注意力頭數)
- `mask_ratio`: 0.15 (mask比例)
- `learning_rate`: 0.001 (適合簡化模型)
- `batch_size`: 64 
- `window_size`: 10 (樣本文本長度)

## 核心組件說明

### SimpleBertModel類
```python
class SimpleBertModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=8):
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # 位置編碼
        self.position_embedding = nn.Embedding(512, hidden_size)
        # Transformer編碼器
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 語言模型頭部
        self.lm_head = nn.Linear(hidden_size, vocab_size)
```

### 訓練流程
1. **數據準備**: 在`build_dataset`中處理mask
2. **前向傳播**: 詞嵌入 + 位置編碼 + Transformer編碼
3. **損失計算**: 只計算mask位置的損失
4. **反向傳播**: 更新模型參數

## 文件結構
- `bert_masked.py`: 主要的模型實現
- `vocab.txt`: 詞彙表
- `corpus.txt`: 訓練語料
- `requirements.txt`: 依賴包列表

## 技術要點

### 1. 為什麼選擇簡化版本？
- **避免依賴問題**: 不依賴複雜的transformers配置
- **更好的控制**: 可以精確控制每個組件
- **易於調試**: 代碼更清晰，問題更容易定位

### 2. 關鍵設計決策
- **數據準備階段mask**: 避免forward中的inplace操作
- **維度處理**: 正確處理tensor維度匹配
- **類型檢查**: 確保輸入類型的正確性

### 3. 性能優化
- **較小的模型**: 適合快速訓練和測試
- **合理的參數**: 平衡性能和計算資源
- **穩定的訓練**: 避免梯度爆炸和消失

## 測試結果
```
✅ 前向傳播成功，輸出形狀: torch.Size([1, 4, 6])
✅ 損失計算成功，損失值: 1.0101
🎉 簡化BERT模型測試通過！
```
