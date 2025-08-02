import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import random


# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# 自定义数据集类
class MaskedLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

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

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding['token_type_ids'].squeeze()

        # 复制一份原始ID用于标签
        labels = input_ids.clone()

        # 应用masking策略
        for i in range(1, len(input_ids) - 1):  # 跳过[CLS]和[SEP]
            if random.random() < self.mask_prob:
                prob = random.random()

                if prob < 0.8:
                    # 80%的概率替换为[MASK]
                    input_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10%的概率替换为随机token
                    input_ids[i] = random.randint(0, len(self.tokenizer) - 1)
                # 10%的概率保持不变（标签依然是原始token）

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


# 定义BERT+MLM模型
class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_model'])
        self.vocab_size = self.bert.config.vocab_size

        # 定义MLM头部
        self.mlm_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, self.vocab_size)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        logits = self.mlm_head(sequence_output)  # [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }


# 训练函数
def train(model, train_dataloader, optimizer, scheduler, device, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}')

        for step, batch in progress_bar:
            # 将数据加载到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs['loss']
            total_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

        # 打印每个epoch的平均损失
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}')


# 评估函数
def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs['loss']
            total_loss += loss.item()

            # 计算准确率（只考虑mask的token）
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=2)

            # 创建mask以只考虑被mask的位置
            masked_positions = (labels != 0) & (labels != input_ids)

            # 计算正确预测的数量
            correct = ((predictions == labels) & masked_positions).sum().item()
            total_correct += correct

            # 计算总的预测数量
            total_predictions += masked_positions.sum().item()

    # 计算平均损失和准确率
    avg_loss = total_loss / len(eval_dataloader)
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    print(f'Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


# 主函数
def main():
    # 配置参数
    config = {
        'bert_model': 'bert-base-uncased',
        'max_len': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3,
        'warmup_steps': 500,
        'mask_prob': 0.15
    }

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(config['bert_model'])
    model = BertForMaskedLM(config)
    model.to(device)

    # 示例数据 - 实际应用中应替换为真实语料库
    train_texts = [
        "Hello, how are you today?",
        "I am learning about transformers and language models.",
        "Natural language processing is an exciting field.",
        "BERT is a powerful model for various NLP tasks.",
        "Training a language model requires a lot of data."
    ]

    eval_texts = [
        "This is a sample sentence for evaluation.",
        "Masked language modeling helps the model understand context."
    ]

    # 创建数据集和数据加载器
    train_dataset = MaskedLMDataset(train_texts, tokenizer, config['max_len'], config['mask_prob'])
    eval_dataset = MaskedLMDataset(eval_texts, tokenizer, config['max_len'], config['mask_prob'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'])

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_dataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'],
                                                num_training_steps=total_steps)

    # 训练模型
    print("开始训练...")
    train(model, train_dataloader, optimizer, scheduler, device, config['epochs'])

    # 评估模型
    print("开始评估...")
    evaluate(model, eval_dataloader, device)

    # 保存模型
    model_save_path = 'bert_mlm_model'
    model.bert.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型已保存到 {model_save_path}")


if __name__ == "__main__":
    main()
