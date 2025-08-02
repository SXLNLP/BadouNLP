# 第十一週作業：SFT (Supervised Fine-Tuning) 實作

## 📋 作業重點

本週重點是實現 **SFT (Supervised Fine-Tuning)** 訓練，使用 Transformer 架構對預訓練模型進行監督式微調。

## 🎯 主要目標

1. **理解 SFT 訓練流程** - 掌握監督式微調的核心概念
2. **實作 Transformer 模型** - 從零開始構建完整的 Transformer 架構
3. **數據處理與訓練** - 使用 `sample_data.json` 進行問答對訓練
4. **模型評估與測試** - 驗證訓練成果並分析模型效果

## 🏗️ 技術架構

### 核心組件
- **模型架構**: Transformer (Encoder-Decoder)
- **訓練方式**: Supervised Fine-Tuning
- **數據格式**: JSON 問答對 (`{"question": "...", "answer": "..."}`)
- **優化器**: Adam
- **損失函數**: CrossEntropyLoss

### 文件結構
```
week11/
├── main.py              # 主要訓練程式
├── loader.py            # 數據加載與預處理
├── config.py            # 訓練配置參數
├── evaluate.py          # 模型評估
├── transformer/         # Transformer 模型組件
│   ├── Models.py        # 主模型定義
│   ├── Layers.py        # 模型層組件
│   ├── SubLayers.py     # 子層實現
│   ├── Modules.py       # 功能模組
│   ├── Optim.py         # 優化器
│   └── Translator.py    # 推理翻譯器
├── sample_data.json     # 訓練數據
└── vocab.txt           # 詞彙表
```

## 🚀 實作步驟

### 1. 數據準備
- 使用 `sample_data.json` 中的 105 個問答對
- 將 question 作為輸入，answer 作為目標輸出
- 通過詞彙表進行 tokenization

### 2. 模型配置
- **模型參數**: 1層 Transformer，2個注意力頭，128維
- **訓練參數**: 300 epochs，批次大小 32，學習率 1e-3
- **序列長度**: 輸入最大 120 tokens，輸出最大 30 tokens

### 3. 訓練過程
- 實現完整的訓練循環
- 使用 Teacher Forcing 策略
- 每個 epoch 後保存模型權重
- 實時監控損失變化

### 4. 模型評估
- 使用 Beam Search 進行推理
- 計算生成文本的連貫性和多樣性
- 分析模型學習效果

## 📊 訓練結果

### 成功指標
- ✅ **模型成功訓練**: 完成 300 輪訓練
- ✅ **損失穩定下降**: 從初始 8.596 降至 5.020
- ✅ **中文生成能力**: 學會生成中文字符
- ✅ **基本架構正常**: Transformer 模型運行穩定

## 🛠️ 使用方法

### 訓練模型
```bash
python main.py
```

### 測試模型
```bash
python test_trained_model.py
```

## 📚 學習重點

1. **SFT 核心概念**: 理解監督式微調與預訓練的區別
2. **Transformer 架構**: 掌握注意力機制和編碼器-解碼器結構
3. **數據處理流程**: 從原始文本到模型輸入的完整流程
4. **訓練策略**: Teacher Forcing、Beam Search 等技術
5. **模型評估**: 如何客觀評估生成模型的質量

## 🔍 技術細節

- **詞彙表大小**: 6219 個 tokens
- **模型保存**: 每輪訓練後保存到 `output/` 目錄
- **推理方式**: Beam Search (beam_size=5)
- **特殊標記**: [PAD]=0, [UNK]=1, [START]=2, [END]=3