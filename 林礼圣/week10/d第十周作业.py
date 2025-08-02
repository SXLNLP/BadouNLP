#week10作业
#使用Bert+mask做自回归语言模型训练。

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# 创建因果掩码（下三角矩阵）
def create_causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0)  # [1, size, size]

# 修改BERT模型实现自回归
class AutoRegressiveBert(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            is_decoder=True  # 关键设置
        )
        self.bert = BertForMaskedLM(config)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask, labels=None):
        seq_len = input_ids.size(1)
        causal_mask = create_causal_mask(seq_len).to(input_ids.device)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=causal_mask
        )
        return outputs.loss if labels is not None else outputs.logits

# 训练函数
def train(model, dataloader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            # 创建标签：输入右移一位
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:].clone()
            labels[:, -1] = -100  # 忽略最后一个token
            
            optimizer.zero_grad()
            loss = model(input_ids, mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}')

# 主程序
if __name__ == "__main__":
    # 示例数据
    texts = [
        "深度学习是人工智能的重要分支",
        "Transformer模型改变了NLP领域",
        "自回归语言模型常用于文本生成"
    ] * 100  # 扩展数据集
    
    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = TextDataset(texts, tokenizer, max_len=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建模型
    model = AutoRegressiveBert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 开始训练
    train(model, dataloader, optimizer, epochs=10)
    
    # 示例生成
    def generate_text(model, prompt, max_length=20):
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = model(input_ids, attention_mask=None)
                next_token = logits[0, -1].argmax().unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == tokenizer.sep_token_id:
                    break
        
        return tokenizer.decode(input_ids[0])
    
    print("\nGenerated Text:", generate_text(model, "人工智能"))
