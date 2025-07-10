import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
import re
import string
from collections import Counter
from tqdm import tqdm
import jieba

# 设置随机种子确保可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
df = pd.read_csv('文本分类练习.csv', header=None, names=['label', 'text'])
print(f"数据集大小: {len(df)}")
print(f"好评数量(1): {sum(df.label == 1)}")
print(f"差评数量(0): {sum(df.label == 0)}")

# 划分训练集和测试集（正负样本各取10%作为测试集）
train_df = pd.DataFrame()
test_df = pd.DataFrame()

for label in [0, 1]:
    subset = df[df['label'] == label]
    train_sub, test_sub = train_test_split(
        subset, test_size=0.1, random_state=SEED
    )
    train_df = pd.concat([train_df, train_sub])
    test_df = pd.concat([test_df, test_sub])

print(f"\n训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

def simple_tokenizer(text):
    # 中文分词
    return list(jieba.cut(text))


# 构建词汇表
def build_vocab(texts, min_freq=5, max_size=10000):
    counter = Counter()
    for text in texts:
        tokens = simple_tokenizer(text)
        counter.update(tokens)

    # 保留最常见的词
    words = [word for word, count in counter.most_common(max_size) if count >= min_freq]

    # 添加特殊标记
    vocab = {'<pad>': 0, '<unk>': 1}
    for i, word in enumerate(words, start=2):
        if i >= max_size:
            break
        vocab[word] = i

    return vocab


vocab = build_vocab(train_df['text'])
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")


def text_to_indices(text):
    tokens = simple_tokenizer(text)
    return [vocab.get(token, vocab['<unk>']) for token in tokens]


label_pipeline = lambda x: int(x)


# 定义Dataset类
class TextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        return text_to_indices(text), label_pipeline(label)


def collate_batch(batch, max_len=128):
    text_list, label_list = [], []
    for (text, label) in batch:
        text_list.append(text[:max_len])
        label_list.append(label)

    padded_text = [t + [0] * (max_len - len(t)) if len(t) < max_len else t[:max_len]
                   for t in text_list]
    return torch.tensor(padded_text, dtype=torch.long), torch.tensor(label_list, dtype=torch.long)


# 创建数据加载器
BATCH_SIZE = 32

train_dataset = TextDataset(train_df)
test_dataset = TextDataset(test_df)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_batch
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_batch
)


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden)


# 定义CNN模型
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100,
                 kernel_sizes=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# 定义BERT模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)


# 训练和评估函数
def train_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs, model_name, bert=False):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{num_epochs}"):
            if bert:
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            else:
                texts, labels = batch
                texts = texts.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if scheduler:
                scheduler.step()

        # 评估
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                if bert:
                    inputs, labels = batch
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = labels.to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    texts, labels = batch
                    texts = texts.to(device)
                    labels = labels.to(device)
                    outputs = model(texts)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch + 1:02} | Loss: {total_loss / len(train_loader):.3f} | '
              f'Acc: {acc:.3f} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')

        if acc > best_acc:
            best_acc = acc

    return best_acc


# BERT需要特殊处理的分词函数
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenizer_batch(text_list, max_length=128, padding='max_length', truncation=True, return_tensors='pt'):
    encoding = bert_tokenizer.batch_encode_plus(
        text_list,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }


# 为BERT创建单独的数据加载器
class BERTDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        return text, int(label)


def bert_collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs = tokenizer_batch(texts)
    return inputs, torch.tensor(labels, dtype=torch.long)


# 创建BERT数据加载器
bert_train_dataset = BERTDataset(train_df)
bert_test_dataset = BERTDataset(test_df)

bert_train_loader = DataLoader(
    bert_train_dataset, batch_size=16,
    shuffle=True, collate_fn=bert_collate_fn
)

bert_test_loader = DataLoader(
    bert_test_dataset, batch_size=16,
    shuffle=False, collate_fn=bert_collate_fn
)

# 定义模型参数
EMBED_DIM = 100
HIDDEN_SIZE = 128
NUM_CLASSES = 2
NUM_EPOCHS = 3  # 减少epoch数以节省时间

# 模型配置列表
model_configs = [
    {
        'name': 'LSTM',
        'model': LSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES),
        'loader': (train_loader, test_loader),
        'optimizer': lambda m: AdamW(m.parameters(), lr=0.001),
        'scheduler': None,
        'hidden_size': HIDDEN_SIZE * 2,
        'bert': False
    },
    {
        'name': 'CNN',
        'model': CNNClassifier(vocab_size, EMBED_DIM, NUM_CLASSES),
        'loader': (train_loader, test_loader),
        'optimizer': lambda m: AdamW(m.parameters(), lr=0.001),
        'scheduler': None,
        'hidden_size': 300,  # 3个卷积层 * 100个滤波器
        'bert': False
    },
    {
        'name': 'BERT',
        'model': BERTClassifier(),
        'loader': (bert_train_loader, bert_test_loader),
        'optimizer': lambda m: AdamW(m.parameters(), lr=2e-5),
        'scheduler': lambda opt: get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=len(bert_train_loader) * NUM_EPOCHS
        ),
        'hidden_size': 768,  # BERT基础版的隐藏大小
        'bert': True
    }
]

# 训练和评估所有模型
results = []

for config in model_configs:
    print(f"\n{'=' * 50}")
    print(f"Training {config['name']} model...")

    model = config['model']
    train_ldr, test_ldr = config['loader']
    optimizer = config['optimizer'](model)
    scheduler = config['scheduler'](optimizer) if config['scheduler'] else None

    # 获取学习率
    lr = optimizer.param_groups[0]['lr']
    hidden_size = config['hidden_size']
    print(f"Learning Rate: {lr}, Hidden Size: {hidden_size}")

    acc = train_model(
        model, train_ldr, test_ldr,
        optimizer, scheduler, NUM_EPOCHS,
        config['name'], config['bert']
    )

    results.append({
        'Model': config['name'],
        'Learning_Rate': lr,
        'Hidden_size': hidden_size,
        'acc': acc
    })

    print(f"{config['name']} model training completed. Test Acc: {acc:.4f}")

# 显示结果表格
print("\n\nResults Summary:")
print("=" * 50)
result_df = pd.DataFrame(results)
print(result_df[['Model', 'Learning_Rate', 'Hidden_size', 'acc']])
print("=" * 50)

# 保存结果到CSV
result_df.to_csv('model_comparison_results.csv', index=False)
print("Results saved to 'model_comparison_results.csv'")
