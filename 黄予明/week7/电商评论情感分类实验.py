import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 读取数据
df = pd.read_csv('文本分类练习数据集/文本分类练习.csv')

# 加载BERT模型和tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese')
bert_model = BertModel.from_pretrained('/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese')
bert_model.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
bert_model = bert_model.to(device)

def get_bert_embeddings(texts, max_len=64):
    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = bert_model(**inputs)
            last_hidden_state = outputs[0]  # 兼容所有transformers版本
            cls_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu()
            all_embeddings.append(cls_embedding)
    return torch.stack(all_embeddings)

# 得到所有样本的BERT [CLS] embedding
X_bert = get_bert_embeddings(df['review'].astype(str).tolist(), max_len=64)
y = torch.tensor(df['label'].tolist())

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42)

# 构建Dataset和DataLoader  
from torch.utils.data import Dataset, DataLoader

class BertFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = BertFeatureDataset(X_train, y_train)
test_ds = BertFeatureDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# 构建CNN模型 (适用于[CLS] embedding)
class CNNClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        # x: (batch, hidden_size) - [CLS] embedding
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
    
# 构建LSTM模型 (适用于[CLS] embedding)
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        # x: (batch, hidden_size) - [CLS] embedding
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
  
# 训练模型与评估
def train_model(model, train_loader, test_loader, epochs=3):
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    # 测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1: {f1_score(all_labels, all_preds):.4f}")

#运行
print("BERT + CNN:")
cnn = CNNClassifier(hidden_size=768, num_classes=2)
train_model(cnn, train_loader, test_loader)

print("BERT + LSTM:")
lstm = LSTMClassifier(hidden_size=768, num_classes=2)
train_model(lstm, train_loader, test_loader)

# 手动输入评论测试
def predict_single(model, text):
    model.eval()
    with torch.no_grad():
        # 获取BERT [CLS] embedding
        inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        last_hidden_state = outputs[0]
        cls_embedding = last_hidden_state[:, 0, :].cpu()
        # 送入分类器
        pred = model(cls_embedding)
        pred_label = pred.argmax(dim=1).item()
        return pred_label

# 选择模型（如用CNN）
model = cnn  # 或 model = lstm

while True:
    user_input = input("请输入一条评论（输入exit退出）：")
    if user_input.lower() == "exit":
        break
    label = predict_single(model, user_input)
    print("预测情感类别：", "正面" if label == 1 else "负面")
