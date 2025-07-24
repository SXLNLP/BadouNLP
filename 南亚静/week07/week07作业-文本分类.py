import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import jieba  # 中文分词，如果是英文数据集则使用nltk
from collections import Counter
import matplotlib.pyplot as plt

# 1. 加载数据并探索
try:
    df = pd.read_csv('文本分类练习.csv', encoding='utf-8')  # 根据实际文件编码调整, 如'gbk'
except:
    # 尝试常见编码
    try:
        df = pd.read_csv('文本分类练习.csv', encoding='gbk')
    except Exception as e:
        print(f"读取文件失败，请检查路径和编码: {e}")
        exit(1)

print("数据预览:")
print(df.head())
print("\n数据信息:")
print(df.info())
print("\n标签分布:")
print(df['label'].value_counts())  # 假设标签列名为'label'，根据实际修改

# 2. 文本预处理与向量化
def preprocess_text(text):
    """清洗文本并分词（针对中文示例）"""
    if not isinstance(text, str):  # 处理可能的NaN或其他非字符串
        return ""
    # 清洗：去除特殊符号、网址等（根据数据集特性增减）
    text = re.sub(r'[^\w\s]', '', text)  # 移除非字母数字非空格字符
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
    # 中文分词 (如果是英文，考虑用nltk.word_tokenize或直接split)
    words = jieba.lcut(text)
    # 可以在此添加停用词过滤 (建议加载停用词表)
    return words

# 对评论列应用预处理 (假设评论列名为'review'，根据实际修改)
df['processed_review'] = df['review'].apply(preprocess_text)

# 构建词汇表
all_words = [word for words in df['processed_review'] for word in words]
word_counts = Counter(all_words)
vocab = {word: idx + 2 for idx, (word, count) in enumerate(word_counts.most_common())}  # idx+2 预留0给PAD, 1给UNK
vocab_size = len(vocab) + 2  # +2 给PAD和UNK
print(f"词汇表大小: {vocab_size}")

# 将单词序列转换为索引序列，并进行填充/截断
def text_to_sequence(words, vocab, max_len):
    seq = [vocab.get(word, 1) for word in words]  # 1代表UNK
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))  # 0代表PAD
    else:
        seq = seq[:max_len]  # 截断
    return seq

# 确定最大序列长度 (可基于分位数)
max_length = 128  # 根据数据集分布调整这个值，可以用 df['processed_review'].apply(len).quantile(0.95)
df['sequence'] = df['processed_review'].apply(lambda x: text_to_sequence(x, vocab, max_length))

# 准备标签 (假设类别是整数，如0,1,2...)
labels = df['label'].values  # 假设标签列名为'label'
num_classes = len(np.unique(labels))  # 类别数
print(f"类别数: {num_classes}")

# 3. 构建PyTorch Dataset & DataLoader
# 将数据和标签转换为Tensor
sequences_tensor = torch.tensor(np.stack(df['sequence'].values), dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)  # 如果标签是类别索引

# 划分训练集、验证集、测试集 (7:2:1)
train_seq, test_seq, train_lbl, test_lbl = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, random_state=42)
train_seq, val_seq, train_lbl, val_lbl = train_test_split(train_seq, train_lbl, test_size=0.125, random_state=42)  # 0.125*0.8=0.1

# 创建TensorDataset和DataLoader
BATCH_SIZE = 64
train_dataset = TensorDataset(train_seq, train_lbl)
val_dataset = TensorDataset(val_seq, val_lbl)
test_dataset = TensorDataset(test_seq, test_lbl)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. 定义模型 (示例：一个简单的嵌入+GRU)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)  # 可尝试bidirectional=True
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        output, hidden = self.rnn(embedded)  # output: (batch, seq_len, hidden_dim), hidden: (num_layers, batch, hidden_dim)
        # 取最后一个时间步的隐藏状态 (双向RNN时需处理)
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (batch_size, num_classes)
        return logits

# 初始化模型
EMBED_DIM = 100  # 词向量维度
HIDDEN_DIM = 128  # RNN隐藏层维度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = TextClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes).to(device)

# 5. 设置训练参数与训练循环
LEARNING_RATE = 0.001
NUM_EPOCHS = 10  # 根据情况调整

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 记录训练指标
train_losses = []
val_losses = []
val_accs = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    running_train_loss = 0.0
    for batch_seq, batch_lbl in train_loader:
        batch_seq, batch_lbl = batch_seq.to(device), batch_lbl.to(device)

        optimizer.zero_grad()
        outputs = model(batch_seq)
        loss = criterion(outputs, batch_lbl)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch_seq.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    running_val_corrects = 0
    with torch.no_grad():
        for batch_seq, batch_lbl in val_loader:
            batch_seq, batch_lbl = batch_seq.to(device), batch_lbl.to(device)

            outputs = model(batch_seq)
            loss = criterion(outputs, batch_lbl)
            _, preds = torch.max(outputs, 1)

            running_val_loss += loss.item() * batch_seq.size(0)
            running_val_corrects += torch.sum(preds == batch_lbl.data)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = running_val_corrects.double() / len(val_loader.dataset)

    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc.item())  # .item() 将tensor标量转成Python数值

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}: ')
    print(f'\t训练损失: {epoch_train_loss:.4f}, 验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.4f}')

    # 保存最佳模型
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        best_model_state = model.state_dict().copy()
        print(f'==> 发现新的最佳验证准确率模型，保存...')

# 加载最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print('加载最佳模型用于测试。')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='训练损失')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='验证损失')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('训练 & 验证损失')
plt.legend()
plt.show()

# 6. 在测试集上评估最佳模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_seq, batch_lbl in test_loader:
        batch_seq, batch_lbl = batch_seq.to(device), batch_lbl.to(device)
        outputs = model(batch_seq)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_lbl.cpu().numpy())

# 计算测试集指标
test_accuracy = accuracy_score(all_labels, all_preds)
print(f'测试集准确率: {test_accuracy:.4f}')
print('\n分类报告:')
print(classification_report(all_labels, all_preds, target_names=[f'Class_{i}' for i in range(num_classes)]))  # 替换为实际类名

# 7. (可选) 对新评论进行预测
def predict_new_review(text, model, vocab, max_length, device):
    """预测一个新评论的类别"""
    model.eval()
    words = preprocess_text(text)
    sequence = text_to_sequence(words, vocab, max_length)
    input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)  # 得到概率分布
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item(), probabilities.squeeze().cpu().numpy()  # 返回类别索引和各类别概率

# 示例使用预测函数
new_review = "这个商品真的太棒了，质量超级好，物流也很快，下次还会购买！"  # 假设这是个好评
predicted_class, class_probs = predict_new_review(new_review, model, vocab, max_length, device)
print(f"\n新评论预测结果: 类别 {predicted_class}")
print(f"各类别概率: {class_probs}")

# 保存模型 (可选)
# torch.save(model.state_dict(), 'text_classifier_model.pth')
