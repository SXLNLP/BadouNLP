# 小米公司知識圖譜問答系統

基於Neo4j的小米公司智能問答系統，支持自然語言查詢企業信息。

## 快速開始

### 1. 安裝依賴
```bash
pip install py2neo pandas openpyxl
```

### 2. 安裝並啟動Neo4j
```bash
# 下載並安裝Neo4j Desktop
# 訪問: https://neo4j.com/download/

# 啟動Neo4j服務
neo4j start
```

### 3. 構建圖譜
```bash
python create_complete_xiaomi_graph.py
```

### 4. 運行問答系統
```bash
python xiaomi_graph_qa_neo4j.py
```

## 支持的問題類型

- **創始人**: "小米集團的創始人是誰"
- **總部**: "小米集團的總部在哪裡"
- **業務**: "小米集團的主要業務是什麼"
- **產品**: "小米集團生產什麼產品"
- **股票**: "小米集團的股票代碼是多少"
- **關係**: "雷軍和小米集團是什麼關係"

## 文件結構

```
├── create_complete_xiaomi_graph.py  # 圖譜構建
├── xiaomi_graph_qa_neo4j.py        # Neo4j問答系統
├── xiaomi_graph_qa_final.py        # 離線問答系統
├── xiaomi_question_templet.xlsx    # 問答模板
├── xiaomi_triplets_*.txt           # 三元組數據
└── xiaomi_kg_schema.json           # 圖譜結構
```

## 系統特點

- 🚀 **雙模式**: 支持Neo4j在線查詢和離線查詢
- 🧠 **智能匹配**: 基於模板的問答匹配
- 🇨🇳 **中文優化**: 專門針對中文企業資料
- 🔍 **實體識別**: 自動識別問題中的實體和關係

## 瀏覽圖譜

訪問Neo4j瀏覽器：`http://localhost:7474`
- 用戶名：`neo4j`
- 密碼：`admin852`