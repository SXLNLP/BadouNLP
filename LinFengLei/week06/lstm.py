import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子确保可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. 数据加载与预处理
# 假设我们有CSV文件，包含'text'和'label'列 (0=负面, 1=正面)
# 这里我们使用模拟数据，实际应用中请替换为真实数据



df = pd.read_csv('your_dataset.csv')

print("数据集示例:")
print(df.head())


# 2. 文本预处理
def preprocess_text(text):
    # 使用jieba进行中文分词
    words = jieba.lcut(text)
    # 移除停用词 (这里简化处理，实际应使用停用词表)
    stopwords = ['，', '。', '！', '？', '、', '的', '了', '是', '就', '也', '在', '和', '有', '这', '那']
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


# 应用预处理
df['processed_text'] = df['text'].apply(preprocess_text)
print("\n预处理后的文本示例:")
print(df[['text', 'processed_text']].head())

# 3. 数据集划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
print(f"\n训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

# 4. 使用BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 5. 创建PyTorch数据集
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 创建数据集和数据加载器
train_dataset = ReviewDataset(
    train_df['processed_text'].values,
    train_df['label'].values,
    tokenizer
)

test_dataset = ReviewDataset(
    test_df['processed_text'].values,
    test_df['label'].values,
    tokenizer
)

BATCH_SIZE = 2  # 小批量适合小数据集，实际应用中可增大
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# 6. 构建BERT分类模型
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)


model = SentimentClassifier().to(device)
print("\n模型结构:")
print(model)

# 7. 训练配置
EPOCHS = 10
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)


# 8. 训练函数
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)
    return avg_loss, accuracy


# 9. 评估函数
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)
    return avg_loss, accuracy, all_preds, all_labels


# 10. 训练循环
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n开始训练...")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print('-' * 10)

    train_loss, train_acc = train_epoch(
        model, train_loader, loss_fn, optimizer, device
    )

    val_loss, val_acc, _, _ = eval_model(
        model, test_loader, loss_fn, device
    )

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')
    print(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')

# 11. 最终评估
print("\n最终测试集评估:")
_, test_acc, y_pred, y_true = eval_model(model, test_loader, loss_fn, device)
print(f'测试集准确率: {test_acc:.4f}')

# 分类报告
print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=['负面', '正面']))

# 混淆矩阵
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'],
            yticklabels=['负面', '正面'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()


# 12. 预测函数
def predict_sentiment(text, model, tokenizer, device, max_len=128):
    model.eval()

    # 预处理文本
    processed_text = preprocess_text(text)

    # 编码文本
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs, dim=1)

    return '正面' if prediction.item() == 1 else '负面'


# 测试预测
test_texts = [
    '这个产品太棒了，强烈推荐！',
    '质量很差，完全不值这个价钱',
    '物流速度快，包装完好',
    '客服态度恶劣，解决问题效率低'
]

print("\n预测示例:")
for text in test_texts:
    sentiment = predict_sentiment(text, model, tokenizer, device)
    print(f"文本: '{text}' -> 情感: {sentiment}")

# 13. 保存模型
torch.save(model.state_dict(), 'sentiment_classifier.pth')
print("\n模型已保存为 'sentiment_classifier.pth'")
