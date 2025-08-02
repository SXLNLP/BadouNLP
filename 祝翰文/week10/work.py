import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
from torch.optim import AdamW

#  配置单向BERT模型
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    is_decoder=True,
    add_cross_attention=False
)
model = BertForMaskedLM(config)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 2. 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()  # 复制作为标签
        }


# 数据
texts = ["Hello, how are you?", "I'm learning about BERT models."]
dataset = TextDataset(texts, tokenizer)

# 加载数据
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 训练环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练过程
model.train()
for epoch in range(3):
    for batch in loader:
        batch_size, seq_len = batch['input_ids'].shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)

        # 准备输入
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device),
            'attention_mask': mask.to(device)  # 覆盖为自回归掩码
        }

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 生成
def generate_text(prompt, max_length=30):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    for _ in range(max_length):
        # 创建自回归掩码
        mask = torch.tril(torch.ones(input_ids.size(1), input_ids.size(1)))
        mask = mask.expand(1, 1, -1, -1).to(device)

        outputs = model(input_ids, attention_mask=mask)
        next_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token == tokenizer.sep_token_id:
            break

    return tokenizer.decode(input_ids[0])


# 测试生成
print(generate_text("The future of AI is"))
