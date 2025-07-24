import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from transformers import BertModel, BertTokenizer
import re
import jieba
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 设置随机种子以确保结果可复现
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# 读取数据 - 修改为处理带引号的CSV格式
def load_data(file_path):
    # 使用csv模块处理带引号的评论
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 查找第一个逗号，区分label和review
            first_comma_idx = line.find(',')
            if first_comma_idx == -1:
                continue

            label = int(line[:first_comma_idx])
            review = line[first_comma_idx + 1:].strip()

            # 处理带引号的评论
            if review.startswith('"') and review.endswith('"'):
                review = review[1:-1]

            data.append({'label': label, 'review': review})

    return pd.DataFrame(data)


# 数据分析
def analyze_data(df):
    # 正负样本数
    pos_samples = df[df['label'] == 1].shape[0]
    neg_samples = df[df['label'] == 0].shape[0]

    # 计算文本长度
    df['text_length'] = df['review'].apply(lambda x: len(str(x)))
    avg_length = df['text_length'].mean()
    max_length = df['text_length'].max()
    min_length = df['text_length'].min()

    # 打印分析结果
    print(f"数据总数: {df.shape[0]}")
    print(f"正样本数: {pos_samples} ({pos_samples / df.shape[0] * 100:.2f}%)")
    print(f"负样本数: {neg_samples} ({neg_samples / df.shape[0] * 100:.2f}%)")
    print(f"文本平均长度: {avg_length:.2f}")
    print(f"文本最大长度: {max_length}")
    print(f"文本最小长度: {min_length}")

    # 绘制文本长度分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['text_length'], bins=50, alpha=0.7, color='skyblue')
    plt.title('文本长度分布')
    plt.xlabel('文本长度')
    plt.ylabel('样本数')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('text_length_distribution.png')
    plt.close()

    return df


# 数据预处理
def preprocess_data(df):
    # 去除特殊字符
    df['clean_review'] = df['review'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(x)))

    # 使用jieba分词
    df['tokenized_review'] = df['clean_review'].apply(lambda x: jieba.lcut(x))

    return df


# 创建词汇表
def build_vocab(df, max_size=10000):
    all_words = []
    for tokens in df['tokenized_review']:
        all_words.extend(tokens)

    # 统计词频
    word_counts = Counter(all_words)

    # 按词频排序
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # 选取前max_size个词作为词汇表
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in sorted_words[:max_size - 2]:
        vocab[word] = len(vocab)

    return vocab


# 创建数据集
class SentimentDataset(Dataset):
    def __init__(self, reviews, labels, vocab):
        self.reviews = reviews
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # 将文本转换为索引
        indices =  [self.vocab.get(word, self.vocab['<unk>']) for word in review]

        return {
            'review': review,
            'indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 自定义collate_fn函数，处理变长序列
def collate_fn(batch):
    # 提取批次中的数据
    reviews = [item['review'] for item in batch]
    indices = [item['indices'] for item in batch]
    labels = [item['label'] for item in batch]

    # 对序列进行填充
    indices = pad_sequence(indices, batch_first=True, padding_value=0)

    # 将标签转换为张量
    labels = torch.stack(labels)

    return {
        'review': reviews,
        'indices': indices,
        'label': labels
    }


# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.dropout(self.embedding(text))

        # embedded = [batch size, sent len, emb dim]

        # LSTM需要三个输入：输入，隐藏状态和单元状态
        # 我们只关心输出和最终隐藏状态
        output, (hidden, cell) = self.lstm(embedded)

        # output = [batch size, sent len, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # 我们只需要最终时间步的隐藏状态
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


# TextRNN模型（使用GRU）
class TextRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.dropout(self.embedding(text))

        # embedded = [batch size, sent len, emb dim]

        output, hidden = self.gru(embedded)

        # output = [batch size, sent len, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]

        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


# Gated CNN模型
class GatedCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # 创建多个不同大小的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # 门控机制的卷积层
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = []
        for i, conv in enumerate(self.convs):
            # 应用门控机制
            gate = torch.sigmoid(self.gate_convs[i](embedded))
            conv_output = torch.tanh(conv(embedded))
            gated_output = conv_output * gate

            # gated_output = [batch size, n_filters, sent len - filter_sizes[i] + 1, 1]

            gated_output = gated_output.squeeze(3)

            # gated_output = [batch size, n_filters, sent len - filter_sizes[i] + 1]

            pooled = nn.functional.max_pool1d(gated_output, gated_output.shape[2]).squeeze(2)

            # pooled = [batch size, n_filters]

            conved.append(pooled)

        # conved = list of [batch size, n_filters]

        cat = self.dropout(torch.cat(conved, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


# BERT模型
class BERTClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-chinese')

        embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, attention_mask):
        with torch.no_grad():
            embedded = self.bert(input_ids=text, attention_mask=attention_mask)[0]

        _, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        return self.out(hidden)


# 定义BERT的数据集类
class BERTSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# BERT的collate_fn函数
def bert_collate_fn(batch):
    # 提取批次中的数据
    texts = [item['text'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    # 对序列进行填充
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 将标签转换为张量
    labels = torch.stack(labels)

    return {
        'text': texts,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }


# 训练函数
def train_model(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        optimizer.zero_grad()

        if isinstance(model, BERTClassifier):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            predictions = model(input_ids, attention_mask).squeeze(1)
        else:
            indices = batch['indices'].to(device)
            labels = batch['label'].to(device)

            predictions = model(indices).squeeze(1)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 评估函数
def evaluate_model(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in iterator:
            if isinstance(model, BERTClassifier):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                predictions = model(input_ids, attention_mask).squeeze(1)
            else:
                indices = batch['indices'].to(device)
                labels = batch['label'].to(device)

                predictions = model(indices).squeeze(1)

            loss = criterion(predictions, labels)

            epoch_loss += loss.item()

            # 获取预测结果
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    return epoch_loss / len(iterator), accuracy, precision, recall, f1


# 测试模型预测速度
def test_inference_speed(model, iterator, device, model_name):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for batch in iterator:
            if isinstance(model, BERTClassifier):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                model(input_ids, attention_mask)
            else:
                indices = batch['indices'].to(device)

                model(indices)

    end_time = time.time()
    total_time = end_time - start_time
    samples = len(iterator.dataset)
    time_per_sample = total_time / samples * 1000  # 转换为毫秒

    print(f"{model_name} 预测速度: {time_per_sample:.2f} 毫秒/样本")
    return time_per_sample


def main():
    # 文件路径
    file_path = "TextAnalysis.csv"

    # 加载数据
    df = load_data(file_path)

    # 数据分析
    analyzed_df = analyze_data(df)

    # 数据预处理
    processed_df = preprocess_data(analyzed_df)

    # 构建词汇表
    vocab = build_vocab(processed_df)

    # 划分训练集和测试集
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=SEED)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 存储模型结果
    results = []

    # ---------------------
    # LSTM模型
    # ---------------------
    print("\n=== LSTM模型 ===")

    # 创建数据集
    train_dataset = SentimentDataset(train_df['tokenized_review'].tolist(),
                                     train_df['label'].tolist(),
                                     vocab)
    test_dataset = SentimentDataset(test_df['tokenized_review'].tolist(),
                                    test_df['label'].tolist(),
                                    vocab)

    # 创建数据加载器，使用自定义collate_fn
    BATCH_SIZE = 16
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # 模型参数
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = vocab['<pad>']

    # 初始化模型
    lstm_model = LSTMClassifier(VOCAB_SIZE,
                                EMBEDDING_DIM,
                                HIDDEN_DIM,
                                OUTPUT_DIM,
                                N_LAYERS,
                                BIDIRECTIONAL,
                                DROPOUT,
                                PAD_IDX)

    lstm_model = lstm_model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(lstm_model.parameters())
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # 训练模型
    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        train_loss = train_model(lstm_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(lstm_model, test_iterator, criterion, device)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # 测试预测速度
    lstm_speed = test_inference_speed(lstm_model, test_iterator, device, "LSTM")

    # 保存结果
    results.append({
        'model': 'LSTM',
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'inference_time': lstm_speed
    })

    # ---------------------
    # TextRNN模型
    # ---------------------
    print("\n=== TextRNN模型 ===")

    # 初始化模型
    rnn_model = TextRNNClassifier(VOCAB_SIZE,
                                  EMBEDDING_DIM,
                                  HIDDEN_DIM,
                                  OUTPUT_DIM,
                                  N_LAYERS,
                                  BIDIRECTIONAL,
                                  DROPOUT,
                                  PAD_IDX)

    rnn_model = rnn_model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(rnn_model.parameters())

    # 训练模型
    for epoch in range(N_EPOCHS):
        train_loss = train_model(rnn_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(rnn_model, test_iterator, criterion, device)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # 测试预测速度
    rnn_speed = test_inference_speed(rnn_model, test_iterator, device, "TextRNN")

    # 保存结果
    results.append({
        'model': 'TextRNN',
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'inference_time': rnn_speed
    })

    # ---------------------
    # Gated CNN模型
    # ---------------------
    print("\n=== Gated CNN模型 ===")

    # 模型参数
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]

    # 初始化模型
    cnn_model = GatedCNN(VOCAB_SIZE,
                         EMBEDDING_DIM,
                         N_FILTERS,
                         FILTER_SIZES,
                         OUTPUT_DIM,
                         DROPOUT,
                         PAD_IDX)

    cnn_model = cnn_model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(cnn_model.parameters())

    # 训练模型
    for epoch in range(N_EPOCHS):
        train_loss = train_model(cnn_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(cnn_model, test_iterator, criterion, device)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # 测试预测速度
    cnn_speed = test_inference_speed(cnn_model, test_iterator, device, "Gated CNN")

    # 保存结果
    results.append({
        'model': 'Gated CNN',
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'inference_time': cnn_speed
    })

    # ---------------------
    # BERT模型
    # ---------------------
    print("\n=== BERT模型 ===")

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建BERT数据集
    train_dataset_bert = BERTSentimentDataset(train_df['review'].tolist(),
                                              train_df['label'].tolist(),
                                              tokenizer)
    test_dataset_bert = BERTSentimentDataset(test_df['review'].tolist(),
                                             test_df['label'].tolist(),
                                             tokenizer)

    # 创建数据加载器，使用BERT的collate_fn
    BATCH_SIZE = 8  # BERT需要更小的批次大小
    train_iterator_bert = DataLoader(train_dataset_bert, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=bert_collate_fn)
    test_iterator_bert = DataLoader(test_dataset_bert, batch_size=BATCH_SIZE, collate_fn=bert_collate_fn)

    # 模型参数
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25

    # 初始化模型
    bert_model = BERTClassifier(HIDDEN_DIM,
                                OUTPUT_DIM,
                                N_LAYERS,
                                BIDIRECTIONAL,
                                DROPOUT)

    bert_model = bert_model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)

    # 训练模型
    N_EPOCHS = 3  # BERT通常不需要太多轮次
    for epoch in range(N_EPOCHS):
        train_loss = train_model(bert_model, train_iterator_bert, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(bert_model, test_iterator_bert, criterion,
                                                                           device)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    # 测试预测速度
    bert_speed = test_inference_speed(bert_model, test_iterator_bert, device, "BERT")

    # 保存结果
    results.append({
        'model': 'BERT',
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'inference_time': bert_speed
    })

    # ---------------------
    # 结果汇总
    # ---------------------
    print("\n=== 模型对比结果 ===")
    results_df = pd.DataFrame(results)
    print(results_df)

    # 保存结果到CSV
    results_df.to_csv('model_comparison_results.csv', index=False)

    # 绘制对比图表
    plt.figure(figsize=(12, 6))

    # 绘制准确率对比
    plt.subplot(1, 2, 1)
    plt.bar(results_df['model'], results_df['accuracy'], color='skyblue')
    plt.title('模型准确率对比')
    plt.xlabel('模型')
    plt.ylabel('准确率')
    plt.ylim(0, 1)

    # 添加数值标签
    for i, v in enumerate(results_df['accuracy']):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    # 绘制预测速度对比
    plt.subplot(1, 2, 2)
    plt.bar(results_df['model'], results_df['inference_time'], color='lightgreen')
    plt.title('模型预测速度对比 (毫秒/样本)')
    plt.xlabel('模型')
    plt.ylabel('预测时间')

    # 添加数值标签
    for i, v in enumerate(results_df['inference_time']):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()


if __name__ == '__main__':
    main()
