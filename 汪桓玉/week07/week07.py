import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, BertConfig
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Activation, \
    GRU, Bidirectional
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import os
from typing import Dict, Tuple

# 数据预处理
X = data['review'].values
y = data['label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

maxlen = 100
max_words = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

X_train_tensor = torch.tensor(X_train_pad, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_parameters_by_layer(model: torch.nn.Module) -> int:
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: {param.shape} - {param_count:,} 参数")
    
    print(f"\n总参数量: {total_params:,}")
    return total_params


def calculate_bert_components_params(config: BertConfig) -> Dict[str, int]:
    embedding_params = (
        config.vocab_size * config.hidden_size +
        config.max_position_embeddings * config.hidden_size +
        config.type_vocab_size * config.hidden_size
    )
    
    attention_params_per_layer = (
        4 * config.hidden_size * config.hidden_size +
        4 * config.hidden_size
    )
    
    ff_params_per_layer = (
        config.hidden_size * config.intermediate_size +
        config.intermediate_size +
        config.intermediate_size * config.hidden_size +
        config.hidden_size
    )
    
    layer_norm_params = 4 * config.hidden_size
    
    transformer_layer_params = attention_params_per_layer + ff_params_per_layer + layer_norm_params
    total_transformer_params = transformer_layer_params * config.num_hidden_layers
    
    pooler_params = config.hidden_size * config.hidden_size + config.hidden_size
    
    theoretical_total = embedding_params + total_transformer_params + pooler_params
    
    return {
        "embedding": embedding_params,
        "attention_per_layer": attention_params_per_layer,
        "feedforward_per_layer": ff_params_per_layer,
        "layer_norm_per_layer": layer_norm_params,
        "transformer_per_layer": transformer_layer_params,
        "all_transformer_layers": total_transformer_params,
        "pooler": pooler_params,
        "total": theoretical_total
    }


def load_pretrained_bert(model_path: str) -> Tuple[BertModel, int]:
    try:
        print(f"正在从 {model_path} 加载预训练的BERT模型...")
        model = BertModel.from_pretrained(model_path)
        print("\n预训练的BERT模型参数统计:")
        params = print_parameters_by_layer(model)
        return model, params
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        return None, 0


def create_bert_from_config() -> Tuple[BertModel, BertConfig, int]:
    print("\n使用BERT-base配置创建模型...")
    config = BertConfig(
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    
    model = BertModel(config)
    print("\nBERT-base模型参数统计:")
    params = print_parameters_by_layer(model)
    
    return model, config, params


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bert_path = os.path.join(current_dir, "bert-base-chinese")
    
    bert, _ = load_pretrained_bert(bert_path)
    
    model, config, actual_params = create_bert_from_config()
    
    print("\nBERT模型主要组成部分的参数量:")
    component_params = calculate_bert_components_params(config)
    
    for component, params in component_params.items():
        if component == "total":
            continue
        print(f"{component.replace('_', ' ').title()}参数量: {params:,}")
    
    print(f"\n理论计算的总参数量: {component_params['total']:,}")
    print(f"实际统计的总参数量: {actual_params:,}")


if __name__ == "__main__":
    main()