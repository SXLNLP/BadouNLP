# 构造随机包含a的字符串，使用rnn进行多分类，类别为a第一次出现在字符串中的位置。

# 完整简化版代码
import numpy as np
import random
import string
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 数据生成
def generate_string(min_len=5, max_len=10):
    length = random.randint(min_len, max_len)
    other_chars = string.ascii_lowercase.replace('a', '')
    s = ''.join(random.choice(other_chars) for _ in range(length))
    pos = random.randint(0, len(s))
    s = s[:pos] + 'a' + s[pos:]
    first_a = s.index('a')
    return s, first_a

samples, labels = [], []
for _ in range(1000):
    s, pos = generate_string()
    samples.append(s)
    labels.append(pos)

# 2. 数据预处理
chars = sorted(list(set(''.join(samples))))
char_to_num = {c:i for i,c in enumerate(chars)}
num_to_char = {i:c for i,c in enumerate(chars)}
max_len = max(len(s) for s in samples)

X = []
for s in samples:
    encoded = [char_to_num[c] for c in s]
    encoded += [0] * (max_len - len(encoded))
    X.append(encoded)
X = np.array(X)
y = to_categorical(labels, num_classes=max_len)

# 3. 构建模型
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=8, input_length=max_len),
    SimpleRNN(32),
    Dense(max_len, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# 5. 评估
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n测试准确率: {accuracy:.2f}")

# 6. 预测示例
def predict_first_a(model, s):
    encoded = [char_to_num.get(c, 0) for c in s]
    encoded += [0] * (max_len - len(encoded))
    encoded = np.array([encoded])
    pred = model.predict(encoded)
    predicted_pos = np.argmax(pred[0])
    if predicted_pos >= len(s):
        predicted_pos = len(s) - 1
    return predicted_pos

test_strings = ["banana", "apple", "grape", "a", "noahere"]
for s in test_strings:
    pred_pos = predict_first_a(model, s)
    actual_pos = s.index('a') if 'a' in s else -1
    print(f"'{s}': 预测 {pred_pos}, 实际 {actual_pos}{' ✓' if pred_pos == actual_pos else ' ✗'}")

# 7. 可视化
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()
plt.show()
