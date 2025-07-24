import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# 自定义字符级Tokenizer
class CharTokenizer:
    def __init__(self, vocab_path=None):
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for idx, char in enumerate(f):
                    self.vocab[char.strip()] = idx + 5  # 跳过特殊符号
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, chars):
        return chars  # 直接返回字符列表
        
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.vocab['[UNK]']) for t in tokens]
    
    def encode_plus(self, text, max_length=128, padding='max_length', truncation=True):
        tokens = list(text)
        input_ids = [self.vocab['[CLS]']] + self.convert_tokens_to_ids(tokens) + [self.vocab['[SEP]']]
        if truncation:
            input_ids = input_ids[:max_length-1] + [self.vocab['[SEP]']]
        if padding == 'max_length':
            attention_mask = [1] * len(input_ids)
            padding_length = max_length - len(input_ids)
            input_ids += [self.vocab['[PAD]']] * padding_length
            attention_mask += [0] * padding_length
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# 数据集类（修正label2id传递问题）
class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id  # 添加标签映射
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 添加特殊token
        text = ['[CLS]'] + text + ['[SEP]']
        label = ['O'] + label + ['O']
        
        # 截断处理
        if len(text) > self.max_len - 1:
            text = text[:self.max_len-1] + ['[SEP]']
            label = label[:self.max_len-1] + ['O']
        else:
            text[-1] = '[SEP]'
            
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(text)
        label_ids = [self.label2id.get(l, 0) for l in label]  # 使用传入的label2id
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.vocab['[PAD]']] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [0] * padding_length  # 0对应PAD标签
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids)
        }

# 模型定义
class BertForNER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # 加载本地模型配置
        self.config = BertConfig.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", return_dict=False) 
        # 初始化BERT模型
        self.bert = BertModel.from_pretrained(r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese", config=self.config)
        # 分类层
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 修改这里：直接获取outputs的第一个元素（sequence_output）
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # 更改这里获取方式
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标签
            active_logits = logits.view(-1, logits.shape[2])
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
            
        return (loss, logits)

# 数据加载函数
def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        for line in f:
            if line.strip() == '':
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words = []
                    tags = []
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            char, tag = parts[0], parts[1]
            words.append(char)
            tags.append(tag)
    return sentences, labels

# 生成标签映射
def create_label_mapping(all_labels):
    unique_labels = set(tag for tags in all_labels for tag in tags)
    label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

# 数据路径
data_path = "F:\\LeetCoding\\torch\\.idea\\week9 序列标注问题\\ner\\ner_data\\test"
texts, labels = load_data(data_path)

# 创建标签映射
label2id, id2label = create_label_mapping(labels)

# 创建数据集
tokenizer = CharTokenizer(vocab_path='F:/models/bert-base-chinese/vocab.txt')
dataset = NERDataset(texts, labels, tokenizer, label2id)  # 添加label2id参数
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 模型初始化
model = BertForNER(num_labels=len(label2id))
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(dataloader) * 3  # 3 epochs
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss}")

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'label2id': label2id,
    'id2label': id2label
}, "ner_model.pth")
tokenizer.save_pretrained("ner_model")
