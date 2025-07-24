import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 参数设置
vocab_size = 27  # 包括大小写字母和空格，总共26个字母+1个空格+1个未知字符（例如用于填充）
sequence_length = 50  # 序列的最大长度

# 构建模型
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # 为每个时间步输出一个概率值

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 示例文本数据
texts = ["hello world", "this is a test", "another example"]
labels = [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]  # 'a'的位置标记为1，其他为0

# 文本向量化
tokenizer = Tokenizer(num_words=vocab_size - 1)  # -1 因为我们不包括未知字符的索引
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=sequence_length)
y = np.array(labels)  # 直接使用预先定义的标签数组作为目标输出
model.fit(X, y, epochs=10, batch_size=2)
loss, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy}")
