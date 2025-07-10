import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, SimpleRNN, LSTM, Dense, 
                                    Dropout, Input, MultiHeadAttention, 
                                    LayerNormalization, GlobalAveragePooling1D)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig

# 1. 解压BERT模型（如果尚未解压）
bert_zip_path = 'bert_base_chinese.zip'
bert_dir = 'bert_base_chinese'

if not os.path.exists(bert_dir):
    print("正在解压BERT模型...")
    with zipfile.ZipFile(bert_zip_path, 'r') as zip_ref:
        zip_ref.extractall(bert_dir)
    print("解压完成！")

# 2. 加载数据
print("\n正在加载数据...")
df = pd.read_csv('文本分类练习.csv')

# 检查数据
print("\n数据样例:")
print(df.head())
print("\n标签分布:")
print(df['label'].value_counts())

# 3. 通用参数设置
MAX_LEN = 128  # 最大文本长度
BATCH_SIZE = 32
EPOCHS = 3
EMBEDDING_DIM = 64
RNN_UNITS = 64

# 4. 划分数据集
X_train, X_val, y_train, y_val = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

print(f"\n训练集样本数: {len(X_train)}")
print(f"验证集样本数: {len(X_val)}")

# 5. RNN和LSTM的数据预处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

def preprocess_data(texts, labels):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded, np.array(labels)

X_train_seq, y_train_seq = preprocess_data(X_train, y_train)
X_val_seq, y_val_seq = preprocess_data(X_val, y_val)

# 6. RNN模型
def build_rnn_model():
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        SimpleRNN(RNN_UNITS, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("\n训练RNN模型...")
rnn_model = build_rnn_model()
rnn_start = time.time()
rnn_history = rnn_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)
rnn_time = time.time() - rnn_start

# 7. LSTM模型
def build_lstm_model():
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(RNN_UNITS, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

print("\n训练LSTM模型...")
lstm_model = build_lstm_model()
lstm_start = time.time()
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)
lstm_time = time.time() - lstm_start

# 8. 本地BERT模型加载
print("\n加载本地BERT模型...")
bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
bert_config = BertConfig.from_pretrained(bert_dir, num_labels=1)
bert_model = TFBertForSequenceClassification.from_pretrained(bert_dir, config=bert_config)

# BERT数据预处理
def preprocess_bert(texts, labels):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return [np.array(input_ids), np.array(attention_masks)], np.array(labels)

X_train_bert, y_train_bert = preprocess_bert(X_train, y_train)
X_val_bert, y_val_bert = preprocess_bert(X_val, y_val)

# 编译BERT模型
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bert_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 训练BERT
print("\n训练BERT模型...")
bert_start = time.time()
bert_history = bert_model.fit(
    X_train_bert, y_train_bert,
    validation_data=(X_val_bert, y_val_bert),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)
bert_time = time.time() - bert_start

# 9. 模型评估
def evaluate_model(model, x_val, y_val, is_bert=False, num_samples=100):
    # 计算准确率
    if is_bert:
        y_pred = (bert_model.predict(x_val)[0] > 0).astype(int)
    else:
        y_pred = (model.predict(x_val) > 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    
    # 计算预测时间
    start = time.time()
    if is_bert:
        _ = model.predict(x_val[0][:num_samples], x_val[1][:num_samples])
    else:
        _ = model.predict(x_val[:num_samples])
    pred_time = time.time() - start
    
    return acc, pred_time

# 评估各模型
print("\n=== 模型评估结果 ===")

rnn_acc, rnn_pred_time = evaluate_model(rnn_model, X_val_seq, y_val_seq)
print(f"RNN - 准确率: {rnn_acc:.4f}, 预测100条耗时: {rnn_pred_time:.4f}s")

lstm_acc, lstm_pred_time = evaluate_model(lstm_model, X_val_seq, y_val_seq)
print(f"LSTM - 准确率: {lstm_acc:.4f}, 预测100条耗时: {lstm_pred_time:.4f}s")

bert_acc, bert_pred_time = evaluate_model(bert_model, X_val_bert, y_val_bert, is_bert=True)
print(f"BERT - 准确率: {bert_acc:.4f}, 预测100条耗时: {bert_pred_time:.4f}s")

# 10. 结果汇总
results = [
    {
        'Model': 'RNN',
        'Accuracy': rnn_acc,
        'Training_Time': rnn_time,
        'Prediction_Time_100': rnn_pred_time,
        'Parameters': f"Embedding: {EMBEDDING_DIM}, RNN Units: {RNN_UNITS}"
    },
    {
        'Model': 'LSTM',
        'Accuracy': lstm_acc,
        'Training_Time': lstm_time,
        'Prediction_Time_100': lstm_pred_time,
        'Parameters': f"Embedding: {EMBEDDING_DIM}, LSTM Units: {RNN_UNITS}"
    },
    {
        'Model': 'BERT',
        'Accuracy': bert_acc,
        'Training_Time': bert_time,
        'Prediction_Time_100': bert_pred_time,
        'Parameters': 'bert-base-chinese (本地)'
    }
]

# 创建结果表格
results_df = pd.DataFrame(results)
print("\n模型对比结果:")
print(results_df.to_markdown(index=False))

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('模型准确率对比')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Prediction_Time_100', data=results_df)
plt.title('预测100条耗时(s)')

plt.tight_layout()
plt.show()

# 保存结果
results_df.to_csv('模型对比结果.csv', index=False)
print("\n结果已保存到'模型对比结果.csv'")
