import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertLMHeadModel
from torch.optim import AdamW
import numpy as np


# 1. 配置参数
class Config:
    model_name = "bert-base-uncased"
    max_len = 128  # 最大序列长度
    batch_size = 16
    lr = 5e-5
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 创建模拟数据集
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# 示例数据
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers provide state-of-the-art NLP models.",
    "Autoregressive language modeling predicts next tokens."
]

# 3. 初始化组件
tokenizer = BertTokenizer.from_pretrained(Config.model_name)
model = BertLMHeadModel.from_pretrained(
    Config.model_name,
    is_decoder=True,  # 关键：设置为解码器模式
    add_cross_attention=False  # 不需要交叉注意力
).to(Config.device)
optimizer = AdamW(model.parameters(), lr=Config.lr)

# 4. 创建数据加载器
dataset = TextDataset(texts, tokenizer, Config.max_len)
dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)


# 5. 训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 创建因果掩码 (下三角矩阵)
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(
            batch_size, 1, seq_len, seq_len
        ).to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # 标签用于计算LM损失
            decoder_attention_mask=causal_mask  # 关键：应用因果掩码
        )

        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


# 6. 训练循环
for epoch in range(Config.epochs):
    avg_loss = train(model, dataloader, optimizer, Config.device)
    print(f"Epoch {epoch + 1}/{Config.epochs} | Loss: {avg_loss:.4f}")


# 7. 生成文本示例
def generate_text(model, tokenizer, prompt, max_length=30, device=Config.device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_length):
        # 创建因果掩码
        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,seq_len,seq_len]

        outputs = model(
            input_ids=input_ids,
            decoder_attention_mask=causal_mask
        )

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 停止条件：遇到[SEP]或达到最大长度
        if next_token.item() == tokenizer.sep_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# 测试生成
print("\nGenerated Text:")
print(generate_text(model, tokenizer, "The future of AI is"))
