import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, BertConfig
import time
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 模型统一超参数
config = {
    "hidden_size": 768,
    "num_layers": 2,  # 减少层数以加快训练
    "epochs": 12,
    "batch_size": 16,  # 减小batch size
    "learning_rate": 2e-5,
    "num_classes": 2,
    "max_length": 64,
    "vocab_size": 21128  # BERT中文模型的词汇表大小
}


# 加载数据
def load_data(file_path="文本分类练习.csv", test_size=0.1):
    try:
        # 尝试不同编码方式读取文件
        try:
            df = pd.read_csv(file_path, header=None, names=['label', 'text'])
        except:
            df = pd.read_csv(file_path, header=None, names=['label', 'text'], encoding='gbk')

        # 数据清洗
        df = df.dropna()
        df = df[df['text'].notnull()]
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()

        return train_test_split(texts, labels, test_size=test_size, random_state=42)
    except Exception as e:
        print(f"加载数据出错: {e}")
        raise


# 文本 Tokenization
def tokenize(texts, max_len=64):
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return encoded['input_ids'], encoded['attention_mask']
    except Exception as e:
        print(f"Tokenization出错: {e}")
        raise


# 构建数据集
def create_dataloaders(train_texts, train_labels, test_texts, test_labels):
    try:
        # BERT输入
        input_ids_train, attention_mask_train = tokenize(train_texts, config["max_length"])
        input_ids_test, attention_mask_test = tokenize(test_texts, config["max_length"])

        # 创建TensorDataset
        bert_train_dataset = TensorDataset(input_ids_train, attention_mask_train, torch.LongTensor(train_labels))
        bert_test_dataset = TensorDataset(input_ids_test, attention_mask_test, torch.LongTensor(test_labels))

        # 创建DataLoader
        bert_train_loader = DataLoader(bert_train_dataset, batch_size=config["batch_size"], shuffle=True)
        bert_test_loader = DataLoader(bert_test_dataset, batch_size=config["batch_size"])

        # LSTM/CNN使用相同的输入，但不需要attention mask
        lstm_train_loader = DataLoader(
            TensorDataset(input_ids_train, torch.LongTensor(train_labels)),
            batch_size=config["batch_size"],
            shuffle=True
        )
        lstm_test_loader = DataLoader(
            TensorDataset(input_ids_test, torch.LongTensor(test_labels)),
            batch_size=config["batch_size"]
        )

        return bert_train_loader, bert_test_loader, lstm_train_loader, lstm_test_loader
    except Exception as e:
        print(f"创建DataLoader出错: {e}")
        raise


# 修正后的BERT模型
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # 冻结BERT前几层参数
        for param in self.bert.parameters():
            param.requires_grad = False
        # 解冻最后两层
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 兼容不同版本的transformers输出
        if isinstance(outputs, tuple):
            pooled_output = outputs[1]  # 旧版本
        else:
            pooled_output = outputs.pooler_output  # 新版本

        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=21128, embed_dim=128, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM需要乘以2

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        return self.classifier(out)


# CNN模型
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size=21128, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_size, kernel_size=k)
            for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size * len(self.convs), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]

        # 应用多个卷积核并池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        # 拼接所有卷积核的输出
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, model_type='bert'):
    model.train()
    total_loss = 0
    for batch in train_loader:
        try:
            if model_type == 'bert':
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
            else:  # lstm/cnn
                input_ids, labels = [t.to(device) for t in batch]
                optimizer.zero_grad()
                outputs = model(input_ids)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            print(f"训练过程中出错: {e}")
            continue

    return total_loss / len(train_loader)


# 评估函数
def evaluate_model(model, test_loader, model_type='bert', sample_size=100):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                if i * test_loader.batch_size >= sample_size:
                    break

                if model_type == 'bert':
                    input_ids, attention_mask, labels = [t.to(device) for t in batch]
                    outputs = model(input_ids, attention_mask)
                else:  # lstm/cnn
                    input_ids, labels = [t.to(device) for t in batch]
                    outputs = model(input_ids)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"评估过程中出错: {e}")
                continue

    inference_time = (time.time() - start_time) / total * 100  # 100条平均耗时(ms)
    accuracy = correct / total * 100 if total > 0 else 0
    report = classification_report(all_labels, all_preds, target_names=['差评', '好评'], zero_division=0)

    return accuracy, inference_time, report


if __name__ == "__main__":
    try:
        print("正在加载并预处理数据...")
        train_texts, test_texts, train_labels, test_labels = load_data()
        print(f"训练样本数: {len(train_texts)}, 测试样本数: {len(test_texts)}")

        bert_train_loader, bert_test_loader, lstm_train_loader, lstm_test_loader = create_dataloaders(
            train_texts, train_labels, test_texts, test_labels
        )

        results = {}
        criterion = nn.CrossEntropyLoss()

        # 训练和评估BERT模型
        print("\n===== 训练BERT模型 =====")
        bert_model = BERTClassifier(config["hidden_size"], config["num_classes"]).to(device)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, bert_model.parameters()),
            lr=config["learning_rate"],
            eps=1e-8
        )

        for epoch in range(config["epochs"]):
            avg_loss = train_model(bert_model, bert_train_loader, criterion, optimizer, 'bert')
            print(f"Epoch {epoch + 1}/{config['epochs']} | 损失: {avg_loss:.4f}")

        bert_acc, bert_time, bert_report = evaluate_model(bert_model, bert_test_loader, 'bert')
        results["BERT"] = {"acc": bert_acc, "time": bert_time, "report": bert_report}

        # 训练和评估LSTM模型
        print("\n===== 训练LSTM模型 =====")
        lstm_model = LSTMClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["hidden_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"]
        ).to(device)
        optimizer = optim.AdamW(lstm_model.parameters(), lr=1e-3)  # LSTM使用更大的学习率

        for epoch in range(config["epochs"]):
            avg_loss = train_model(lstm_model, lstm_train_loader, criterion, optimizer, 'lstm')
            print(f"Epoch {epoch + 1}/{config['epochs']} | 损失: {avg_loss:.4f}")

        lstm_acc, lstm_time, lstm_report = evaluate_model(lstm_model, lstm_test_loader, 'lstm')
        results["LSTM"] = {"acc": lstm_acc, "time": lstm_time, "report": lstm_report}

        # 训练和评估CNN模型
        print("\n===== 训练CNN模型 =====")
        cnn_model = CNNClassifier(
            vocab_size=config["vocab_size"],
            embed_dim=config["hidden_size"],
            hidden_size=config["hidden_size"],
            num_classes=config["num_classes"]
        ).to(device)
        optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3)  # CNN使用更大的学习率

        for epoch in range(config["epochs"]):
            avg_loss = train_model(cnn_model, lstm_train_loader, criterion, optimizer, 'cnn')
            print(f"Epoch {epoch + 1}/{config['epochs']} | 损失: {avg_loss:.4f}")

        cnn_acc, cnn_time, cnn_report = evaluate_model(cnn_model, lstm_test_loader, 'cnn')
        results["CNN"] = {"acc": cnn_acc, "time": cnn_time, "report": cnn_report}

        # 输出结果比较
        print("\n===== 模型性能比较 =====")
        for name, res in results.items():
            print(f"\n【{name}模型】")
            print(f"准确率: {res['acc']:.2f}%")
            print(f"100条评论预测耗时: {res['time']:.2f}ms")
            print("\n分类报告:")
            print(res['report'])

    except Exception as e:
        print(f"程序运行出错: {e}")
