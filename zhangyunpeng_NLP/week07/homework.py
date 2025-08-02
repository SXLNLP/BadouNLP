
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Activation, \
    GRU, Bidirectional
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# 数据预处理
X = data['review'].values
y = data['label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 超参数设置
maxlen = 100
max_words = 10000
embedding_dim = 100

# 文本向量化
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# 将数据转换为 PyTorch 的 Tensor 格式
X_train_tensor = torch.tensor(X_train_pad, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# 定义 TextRNN 模型
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.gru = nn.GRU(embedding_dim, 64, bidirectional=True)
        self.fc = nn.Linear(64 * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = torch.cat([x[:, 0, :], x[:, -1, :]], dim=1)
        x = self.fc(x)
        return x


# 定义 TextCNN 模型
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=maxlen - 3 + 1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


# 定义 TextRCNN 模型
class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 64, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(2 * 64, 128, kernel_size=3, padding='same')
        self.pool = nn.MaxPool1d(maxlen)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


# 定义 Gated CNN 模型
def gated_cnn():
    K.clear_session()
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Convolution1D(128, 5, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(256, 5, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3))
    model.add(Convolution1D(512, 5, padding='same', activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 训练和评估模型的函数
def train_and_evaluate(model, model_name):
    if model_name == 'Bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)

        train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_encodings = tokenizer(list(X_test), truncation=True, padding=True)
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(y_test)
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        model.train()
        for epoch in range(3):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        start_time = time.time()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        end_time = time.time()
        accuracy = correct / total
        prediction_time = end_time - start_time

    elif model_name == 'Gated CNN':
        model = gated_cnn()
        model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.2)

        start_time = time.time()
        _, accuracy = model.evaluate(X_test_pad, y_test)
        end_time = time.time()
        prediction_time = end_time - start_time

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(3):
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        start_time = time.time()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        end_time = time.time()
        accuracy = correct / total
        prediction_time = end_time - start_time

    return accuracy, prediction_time


# 模型列表
models = [
    (RNNModel(), 'RNN'),
    (LSTMModel(), 'LSTM'),
    (TextRNN(), 'TextRNN'),
    (TextCNN(), 'TextCNN'),
    (TextRCNN(), 'TextRCNN'),
    (None, 'Bert'),
    (None, 'Gated CNN')
]

# 存储结果
results = []

for model, model_name in models:
    accuracy, prediction_time = train_and_evaluate(model, model_name)
    results.append([model_name, accuracy, prediction_time])

# 将结果转换为 DataFrame 并输出
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Prediction Time (s)'])
print(results_df)
