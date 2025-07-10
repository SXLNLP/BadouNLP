import pandas as pd
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


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# 数据读取方法
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue

            first_comma_idx = line.find(',')
            if first_comma_idx == -1:
                continue
            label = int(line[:first_comma_idx])
            review = line[first_comma_idx + 1:].strip()
            if review.startswith('"') and review.endswith('"'):
                review = review[1:-1]
            data.append({'label': label, 'review': review})
    return pd.DataFrame(data)

# 解析数据
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

    return df


# 预处理数据
def preprocess_data(df):
    # 去除特殊字符
    df['clean_review'] = df['review'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(x)))

    # 使用jieba分词
    df['tokenized_review'] = df['clean_review'].apply(lambda x: jieba.lcut(x))

    return df


# 词汇表
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


# 数据集
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
        indices = [self.vocab.get(word, self.vocab['<unk>']) for word in review]
        return {
            'review': review,
            'indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 自定义collate_fn函数，处理变长序列
def collate_fn(batch):
    reviews = [item['review'] for item in batch]
    indices = [item['indices'] for item in batch]
    labels = [item['label'] for item in batch]
    indices = pad_sequence(indices, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {
        'review': reviews,
        'indices': indices,
        'label': labels
    }


# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


# TextRNN模型（使用GRU）
class TextRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


# Gated CNN模型
class GatedCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.gate_convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = []
        for i, conv in enumerate(self.convs):
            gate = torch.sigmoid(self.gate_convs[i](embedded))
            conv_output = torch.tanh(conv(embedded))
            gated_output = conv_output * gate
            pooled = nn.functional.max_pool1d(gated_output.squeeze(3), gated_output.shape[2]).squeeze(2)
            conved.append(pooled)
        cat = self.dropout(torch.cat(conved, dim=1))
        return self.fc(cat)


# BERT模型
class BERTClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True,
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
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# BERT的collate_fn函数
def bert_collate_fn(batch):
    texts = [item['text'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
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
            predictions = model(batch['input_ids'].to(device),
                                batch['attention_mask'].to(device)).squeeze(1)
        else:
            predictions = model(batch['indices'].to(device)).squeeze(1)
        loss = criterion(predictions, batch['label'].to(device))
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
                predictions = model(batch['input_ids'].to(device),
                                    batch['attention_mask'].to(device)).squeeze(1)
            else:
                predictions = model(batch['indices'].to(device)).squeeze(1)
            loss = criterion(predictions, batch['label'].to(device))
            epoch_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
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
                model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            else:
                model(batch['indices'].to(device))
    end_time = time.time()
    time_per_sample = (end_time - start_time) / len(iterator.dataset) * 1000
    print(f"{model_name} 预测速度: {time_per_sample:.2f} 毫秒/样本")
    return time_per_sample


def main():
    # 文件路径
    file_path = "/Users/wanglei/Desktop/ailearn/traindata/文本分类练习.csv"

    # 加载数据
    df = load_data(file_path)
    analyzed_df = analyze_data(df)
    processed_df = preprocess_data(analyzed_df)
    vocab = build_vocab(processed_df)
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    # LSTM模型
    print("\n=== LSTM模型 ===")
    train_dataset = SentimentDataset(train_df['tokenized_review'].tolist(),
                                     train_df['label'].tolist(), vocab)
    test_dataset = SentimentDataset(test_df['tokenized_review'].tolist(),
                                    test_df['label'].tolist(), vocab)
    train_iterator = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    VOCAB_SIZE = len(vocab)
    lstm_model = LSTMClassifier(VOCAB_SIZE, 100, 256, 2, 2, True, 0.5, vocab['<pad>']).to(device)
    optimizer = optim.Adam(lstm_model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(5):
        train_loss = train_model(lstm_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(lstm_model, test_iterator, criterion, device)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    lstm_speed = test_inference_speed(lstm_model, test_iterator, device, "LSTM")
    results.append({
        'model': 'LSTM', 'accuracy': test_acc, 'precision': test_prec,
        'recall': test_rec, 'f1': test_f1, 'inference_time': lstm_speed
    })

    # TextRNN模型
    print("\n=== TextRNN模型 ===")
    rnn_model = TextRNNClassifier(VOCAB_SIZE, 100, 256, 2, 2, True, 0.5, vocab['<pad>']).to(device)
    optimizer = optim.Adam(rnn_model.parameters())

    for epoch in range(5):
        train_loss = train_model(rnn_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(rnn_model, test_iterator, criterion, device)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    rnn_speed = test_inference_speed(rnn_model, test_iterator, device, "TextRNN")
    results.append({
        'model': 'TextRNN', 'accuracy': test_acc, 'precision': test_prec,
        'recall': test_rec, 'f1': test_f1, 'inference_time': rnn_speed
    })

    # Gated CNN模型
    print("\n=== Gated CNN模型 ===")
    cnn_model = GatedCNN(VOCAB_SIZE, 100, 100, [3, 4, 5], 2, 0.5, vocab['<pad>']).to(device)
    optimizer = optim.Adam(cnn_model.parameters())

    for epoch in range(5):
        train_loss = train_model(cnn_model, train_iterator, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(cnn_model, test_iterator, criterion, device)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    cnn_speed = test_inference_speed(cnn_model, test_iterator, device, "Gated CNN")
    results.append({
        'model': 'Gated CNN', 'accuracy': test_acc, 'precision': test_prec,
        'recall': test_rec, 'f1': test_f1, 'inference_time': cnn_speed
    })

    # BERT模型
    print("\n=== BERT模型 ===")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_dataset_bert = BERTSentimentDataset(train_df['review'].tolist(),
                                              train_df['label'].tolist(), tokenizer)
    test_dataset_bert = BERTSentimentDataset(test_df['review'].tolist(),
                                             test_df['label'].tolist(), tokenizer)
    train_iterator_bert = DataLoader(train_dataset_bert, batch_size=8, shuffle=True, collate_fn=bert_collate_fn)
    test_iterator_bert = DataLoader(test_dataset_bert, batch_size=8, collate_fn=bert_collate_fn)

    bert_model = BERTClassifier(256, 2, 2, True, 0.25).to(device)
    optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)

    for epoch in range(3):
        train_loss = train_model(bert_model, train_iterator_bert, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(bert_model, test_iterator_bert, criterion,
                                                                           device)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    bert_speed = test_inference_speed(bert_model, test_iterator_bert, device, "BERT")
    results.append({
        'model': 'BERT', 'accuracy': test_acc, 'precision': test_prec,
        'recall': test_rec, 'f1': test_f1, 'inference_time': bert_speed
    })

    # 结果汇总
    print("\n=== 模型性能对比 ===")
    print("| {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<15} |".format(
        "模型", "准确率", "精确率", "召回率", "F1分数", "预测速度(ms)"))
    print("|{:-<12}|{:-<10}|{:-<10}|{:-<10}|{:-<10}|{:-<17}|".format("", "", "", "", "", ""))

    for result in results:
        print("| {:<10} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<15.2f} |".format(
            result['model'], result['accuracy'], result['precision'],
            result['recall'], result['f1'], result['inference_time']))

    # 保存结果
    pd.DataFrame(results).to_csv('model_comparison_results.csv', index=False)


if __name__ == '__main__':
    main()
