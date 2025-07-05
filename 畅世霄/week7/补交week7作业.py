import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel
import time
import os

# 模型统一超参数
config = {
    "hidden_size": 768,
    "num_layers": 2,  # 减少层数以加快训练
    "epochs": 12,
    "batch_size": 16,  # 减小batch size
    "learning_rate": 1e-5,
    "num_classes": 2,
    "max_length": 64,
    "vocab_size": 21128  # BERT中文模型的词汇表大小
}
# 检查模型路径是否存在
bert_path = r"E:\PycharmProjects\NLPtask\week7   文本分类问题\practice\bert-base-chinese"
if not os.path.exists(bert_path):
    raise FileNotFoundError(f"BERT模型路径不存在: {bert_path}")


# 加载数据
def load_data(file_path="文本分类练习.csv", test_size=0.2):
    df = pd.read_csv(file_path, header=0, names=['label', 'text'])
    df = df.dropna()
    df = df[df['text'].notnull()]
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].tolist()
    return train_test_split(texts, labels, test_size=test_size, random_state=42)

# 文本 Tokenization
def tokenize(texts, max_len=64):
    tokenizer = BertTokenizer.from_pretrained(
        bert_path,
        local_files_only=True
    )
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return encoded['input_ids'], encoded['attention_mask']



# 构建数据集
def create_dataloaders(train_texts, train_labels, test_texts, test_labels):

    # BERT输入
    input_ids_train, attention_mask_train = tokenize(train_texts, config["max_length"])
    input_ids_test, attention_mask_test = tokenize(test_texts, config["max_length"])

    # 创建TensorDataset
    bert_train_dataset = TensorDataset(input_ids_train, attention_mask_train, torch.LongTensor(train_labels))
    bert_test_dataset = TensorDataset(input_ids_test, attention_mask_test, torch.LongTensor(test_labels))

    # 创建DataLoader
    bert_train_loader = DataLoader(bert_train_dataset, batch_size=config["batch_size"], shuffle=True)
    bert_test_loader = DataLoader(bert_test_dataset, batch_size=config["batch_size"])

    # LSTM/CNN使用相同的输入
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


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert_model_name=bert_path,
                 hidden_size=256,
                 num_classes=2,
                 dropout_prob=0.2,
                 freeze_layers=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_model_name,
            local_files_only=True,
            add_pooling_layer=False
        )
        bert_hidden_size = self.bert.config.hidden_size
        self._freeze_bert_layers(freeze_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(bert_hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_classes)
        )
        self._init_weights()

    def _freeze_bert_layers(self, freeze_layers):
        if freeze_layers >= 12:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in self.bert.encoder.layer[freeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
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
    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

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
            bert_model = BERTClassifier(hidden_size=config["hidden_size"], num_classes=config["num_classes"]).to(device)
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

            # [保留LSTM和CNN的训练评估代码...]

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
            raise e  # 重新抛出异常以查看完整堆栈信息
