import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
import json

# 1. 数据准备

def load_data1(data_path='sample_data.json'):
    data =[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                title = item['title']
                content = item['content']
                data.append(title)
                data.append(content)
    
    return data
def load_data2(data_path='sample_data.json'):
    data =[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                title = item['title']
                content = item['content']
                data.append({"title": title, "content": content})  # 用字典包装
    return data

# 2. 数据预处理
class TextPreprocessor:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4
        
    def tokenize(self, text):
        # 简单分词
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点
        return text.lower().split()
        
    def build_vocab(self, texts):
        # 从文本构建词汇表
        word_counts = defaultdict(int)
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] += 1
        
        # 添加到词汇表
        for word in word_counts:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
            
    def text_to_sequence(self, text, max_len=None):
        # 将文本转换为序列
        tokens = self.tokenize(text)
        sequence = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        
        if max_len:
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            else:
                sequence += [self.word2idx["<PAD>"]] * (max_len - len(sequence))
        
        return sequence
    
    def sequence_to_text(self, sequence):
        # 将序列转换为文本
        tokens = [self.idx2word.get(idx, "<UNK>") for idx in sequence]
        # 去除特殊标记
        tokens = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>"]]
        return " ".join(tokens)

# 初始化并构建预处理工具
processor = TextPreprocessor()
all_texts = load_data1()
# all_texts = [item["content"] for item in data] + [item["title"] for item in data]
processor.build_vocab(all_texts)

# 3. 数据集类
class TitleGenerationDataset(Dataset):
    def __init__(self, data, processor, max_content_len=100, max_title_len=20):
        self.data = data
        self.processor = processor
        self.max_content_len = max_content_len
        self.max_title_len = max_title_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理内容
        content_seq = self.processor.text_to_sequence(
            item["content"], 
            max_len=self.max_content_len
        )
        
        # 处理标题，添加开始和结束标记
        title_tokens = self.processor.tokenize(item["title"])
        title_seq = [self.processor.word2idx["<SOS>"]] + \
                   [self.processor.word2idx.get(token, self.processor.word2idx["<UNK>"]) for token in title_tokens] + \
                   [self.processor.word2idx["<EOS>"]]
        
        # 填充标题
        if len(title_seq) < self.max_title_len:
            title_seq += [self.processor.word2idx["<PAD>"]] * (self.max_title_len - len(title_seq))
        else:
            title_seq = title_seq[:self.max_title_len]
            
        # 准备目标序列（用于训练，shift by 1）
        target_seq = title_seq[1:]  # 移除<SOS>
        # 最后一个位置不需要预测，所以截断
        if len(target_seq) >= self.max_title_len:
            target_seq = target_seq[:self.max_title_len-1]
        else:
            target_seq += [self.processor.word2idx["<PAD>"]] * (self.max_title_len - 1 - len(target_seq))
        
        return {
            "content": torch.tensor(content_seq, dtype=torch.long),
            "title_input": torch.tensor(title_seq[:-1], dtype=torch.long),  # 移除最后一个字符作为输入
            "title_target": torch.tensor(target_seq, dtype=torch.long)
        }

data = load_data2()
# 创建数据集和数据加载器
dataset = TitleGenerationDataset(data, processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 定义LSTM模型 - Seq2Seq架构
class LSTMTitleGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 编码器
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 解码器
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim * 2,  # 结合嵌入和编码器输出
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # 隐藏层维度转换（双向到单向）
        self.hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, content, title_input, hidden=None, cell=None):
        # 嵌入层
        content_emb = self.embedding(content)
        title_emb = self.embedding(title_input)
        
        # 编码器
        encoder_out, (encoder_hidden, encoder_cell) = self.encoder_lstm(content_emb)
        
        # 编码器最后一层的输出作为上下文向量
        context = torch.mean(encoder_out, dim=1)  # 全局上下文
        
        # 准备解码器初始状态（将双向LSTM的输出转换为单向）
        if hidden is None or cell is None:
            # 合并双向隐藏状态
            hidden = torch.tanh(self.hidden_proj(torch.cat((encoder_hidden[-2,:,:], encoder_hidden[-1,:,:]), dim=1)))
            cell = torch.tanh(self.cell_proj(torch.cat((encoder_cell[-2,:,:], encoder_cell[-1,:,:]), dim=1)))
            
            # 扩展到多层
            hidden = hidden.unsqueeze(0).repeat(2, 1, 1)  # 2层LSTM
            cell = cell.unsqueeze(0).repeat(2, 1, 1)
        
        # 将上下文向量与解码器输入拼接
        batch_size, seq_len, _ = title_emb.shape
        context_repeated = context.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = torch.cat([title_emb, context_repeated], dim=2)
        
        # 解码器
        decoder_out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
        
        # 输出层
        output = self.fc(decoder_out)
        
        return output, (hidden, cell)

# 5. 初始化模型、损失函数和优化器
vocab_size = processor.vocab_size
model = LSTMTitleGenerator(vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=processor.word2idx["<PAD>"])  # 忽略填充
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 训练模型（SFT监督微调过程）
def train(model, dataloader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            content = batch["content"]
            title_input = batch["title_input"]
            title_target = batch["title_target"]
            
            # 前向传播
            outputs, _ = model(content, title_input)
            
            # 计算损失
            loss = criterion(
                outputs.reshape(-1, vocab_size), 
                title_target.reshape(-1)
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "output/model2.pth")

# 执行训练
train(model, dataloader, criterion, optimizer, epochs=100)

# 7. 生成标题函数
def generate_title(model, processor, content, max_length=20):
    model.eval()
    with torch.no_grad():
        # 处理输入内容
        content_seq = processor.text_to_sequence(content)
        content_tensor = torch.tensor(content_seq, dtype=torch.long).unsqueeze(0)  # 添加批次维度
        
        # 编码器输出
        content_emb = model.embedding(content_tensor)
        encoder_out, (encoder_hidden, encoder_cell) = model.encoder_lstm(content_emb)
        context = torch.mean(encoder_out, dim=1)
        
        # 初始化解码器输入
        decoder_input = torch.tensor([[processor.word2idx["<SOS>"]]], dtype=torch.long)
        hidden = torch.tanh(model.hidden_proj(torch.cat((encoder_hidden[-2,:,:], encoder_hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(model.cell_proj(torch.cat((encoder_cell[-2,:,:], encoder_cell[-1,:,:]), dim=1)))
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        cell = cell.unsqueeze(0).repeat(2, 1, 1)
        
        generated_sequence = []
        
        for _ in range(max_length):
            # 解码器前向传播
            decoder_emb = model.embedding(decoder_input)
            context_repeated = context.unsqueeze(1)
            decoder_input_combined = torch.cat([decoder_emb, context_repeated], dim=2)
            
            output, (hidden, cell) = model.decoder_lstm(decoder_input_combined, (hidden, cell))
            output = model.fc(output)
            
            # 选择概率最高的词
            predicted_idx = torch.argmax(output, dim=2).item()
            generated_sequence.append(predicted_idx)
            
            # 如果生成了结束标记，停止
            if predicted_idx == processor.word2idx["<EOS>"]:
                break
            
            # 更新解码器输入
            decoder_input = torch.tensor([[predicted_idx]], dtype=torch.long)
        
        # 转换为文本
        return processor.sequence_to_text(generated_sequence)

# 8. 测试模型 - 使用示例数据
test_content = "18日北京广电局电影处处长王健表示，北京明年有望从周一到周五每天上午或下午推出统一半价观影时段，还将要求影票上电影时间须为剔除片前广告的时间。首都影院联盟若7成以上会员同意，将实施此优惠政策（京华时报）"
predicted_title = generate_title(model, processor, test_content)

print("输入内容:", test_content)
print("预测标题:", predicted_title)
# print("实际标题:", "阿根廷歹徒抢服装尺码不对拿回店里换")
